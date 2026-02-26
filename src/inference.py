import joblib
import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any

from src.models.stacking import StackingMetaLearner
from src.models.nn import SiameseLSTM
from src.data import Preprocessor, TennisDataset
from src.config import ProjectConfig
from src.logger import get_logger

logger = get_logger(__name__)


class MetaLearnerPipeline:
    def __init__(self, meta_learner: StackingMetaLearner, lstm_model: SiameseLSTM, preprocessor: Preprocessor, config: ProjectConfig):
        self.meta_learner = meta_learner
        self.lstm_model = lstm_model
        self.preprocessor = preprocessor
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model.to(self.device)

    @classmethod
    def load_frozen_model(cls, base_path: Path):
        """
        Loads the production-ready meta-learner, LSTM model, and preprocessor.
        """
        try:
            config_path = Path("configs/config.yaml")
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found at {config_path}")
            
            config = ProjectConfig.load(str(config_path))
            
            # 1. Load Stacking Meta-Learner
            stacking_path = base_path / "stacking" / "meta_learner.joblib"
            meta_learner_data = joblib.load(stacking_path)
            
            meta_learner = StackingMetaLearner(config, base_path)
            
            if isinstance(meta_learner_data, dict):
                meta_learner.meta_model = meta_learner_data.get('model')
                meta_learner.model_names = meta_learner_data.get('features', [])
            else:
                meta_learner.meta_model = meta_learner_data
                logger.warning("Meta-learner artifact is a raw model; model_names may be uninitialized.")

            # 2. Load Preprocessor
            preprocessor_path = base_path / "global_preprocessor.pkl"
            if not preprocessor_path.exists():
                preprocessor_path = base_path / "lstm" / "global_preprocessor.pkl"
                
            preprocessor = Preprocessor(config).load(preprocessor_path)

            # 3. Load Siamese LSTM
            lstm_path = base_path / "lstm" / "best_model.pt"
            seq_dim = len(preprocessor.seq_indices)
            ctx_dim = len(preprocessor.ctx_indices)

            lstm_model = SiameseLSTM(config, seq_dim, ctx_dim)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
            lstm_model.eval()

            return cls(meta_learner, lstm_model, preprocessor, config)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize MetaLearnerPipeline: {str(e)}") from e

    def predict_proba(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Full inference logic: Preprocessing -> LSTM Embeddings -> Stacking Meta-Learner.
        """
        try:
            # Reconstruct the single row dataframe
            import pandas as pd
            df = pd.DataFrame([data])
            
            # Use the preprocessor to encode the row (in "lstm" mode to get sequences)
            # Assuming seq_len is available from config or hardcoded for inference (e.g. 10)
            seq_len = self.config.architecture.seq_len if hasattr(self.config, 'architecture') else 10
            
            ds_lstm = TennisDataset(df, self.preprocessor, mode="lstm", seq_len=seq_len)
            
            # We only have one item
            seq_a, seq_b, ctx, _ = ds_lstm[0]
            
            # Add batch dimension
            seq_a = seq_a.unsqueeze(0).to(self.device)
            seq_b = seq_b.unsqueeze(0).to(self.device)
            ctx = ctx.unsqueeze(0).to(self.device)
            
            # LSTM Inference
            with torch.no_grad():
                lstm_logits = self.lstm_model(seq_a, seq_b, ctx)
                lstm_prob = torch.sigmoid(lstm_logits).cpu().numpy().flatten()[0]
                
            # Stacking Meta Learner Inference
            # Re-create tabular dataset for base tabular models if needed by meta-learner
            base_preds = {"lstm": np.array([lstm_prob])}
            
            # Note: We need to evaluate other models in the stack if the stacker requires them.
            # Assuming the stacking meta learner is configured with access to them or only uses LSTM
            # For this exact implementation, we pass the LSTM prob to the meta learner.
            final_probs = self.meta_learner.predict_proba(base_preds)
            
            p1_win = final_probs[0] # Probability class 1 (Player 1 wins)
            p2_win = 1.0 - p1_win
            
            return {"p1": float(p1_win), "p2": float(p2_win)}
            
        except Exception as e:
            logger.error(f"Error during predict_proba execution: {e}")
            # Fallback for API stability if required, or re-raise
            raise e