import joblib
import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any

from src.models.stacking import StackingMetaLearner
from src.models.nn import SiameseLSTM
from src.data import Preprocessor
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
            
            # Defensive check: handle cases where the object is the raw model vs a metadata dict
            if isinstance(meta_learner_data, dict):
                meta_learner.meta_model = meta_learner_data.get('model')
                meta_learner.model_names = meta_learner_data.get('features', [])
            else:
                # meta_learner_data is the XGBClassifier directly
                meta_learner.meta_model = meta_learner_data
                # Assuming model_names is handled by StackingMetaLearner internal init or hardcoded
                logger.warning("Meta-learner artifact is a raw model; model_names may be uninitialized.")

            # 2. Load Preprocessor
            preprocessor_path = base_path / "global_preprocessor.pkl"
            if not preprocessor_path.exists():
                # Fallback to subdirectory if that's where your build put it
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
            # Re-raise with specific context to aid API.py logging
            raise RuntimeError(f"Failed to initialize MetaLearnerPipeline: {str(e)}") from e

    def predict_proba(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Full inference logic: Preprocessing -> LSTM Embeddings -> Stacking Meta-Learner.
        """
        # Note: Implement the actual tensor conversion and model calls here
        # to replace the dummy dictionary in the placeholder.
        return {"p1": 0.5, "p2": 0.5}