import joblib
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.models.stacking import StackingMetaLearner
from src.models.nn import SiameseLSTM
from src.data import Preprocessor, TennisDataset
from src.config import ProjectConfig

class MetaLearnerPipeline:
    def __init__(self, meta_learner: StackingMetaLearner, lstm_model: SiameseLSTM, preprocessor: Preprocessor, config: ProjectConfig):
        self.meta_learner = meta_learner
        self.lstm_model = lstm_model
        self.preprocessor = preprocessor
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model.to(self.device)

    @classmethod
    def load_frozen_model(cls, base_path: Path = Path("artifacts/prod")):
        """
        Loads the production-ready meta-learner, LSTM model, and preprocessor from a hardcoded path.
        """
        try:
            # Load Meta-Learner
            stacking_path = base_path / "stacking" / "meta_learner.joblib"
            meta_learner_data = joblib.load(stacking_path)
            
            config = ProjectConfig.load() # Load default config
            meta_learner = StackingMetaLearner(config, base_path)
            meta_learner.meta_model = meta_learner_data['model']
            meta_learner.model_names = meta_learner_data['features']

            # Load Siamese LSTM
            lstm_path = base_path / "lstm" / "best_model.pt"
            # TODO: These dimensions are hardcoded based on the training script.
            # A more robust solution would store these in the artifact directory.
            # Assuming seq_matrix shape [num_samples, seq_len, num_features] -> we need num_features
            # Assuming ctx_matrix shape [num_samples, num_features] -> we need num_features
            # From the config, sequence_features has 7 items, and context_features has 15 items
            # The preprocessor will add more features to context.
            # Let's load the preprocessor first.

            preprocessor_path = base_path / "global_preprocessor.pkl"
            preprocessor = Preprocessor(config).load(preprocessor_path)

            # Determine dimensions from the loaded preprocessor
            seq_dim = len(preprocessor.seq_indices)
            ctx_dim = len(preprocessor.ctx_indices)

            if seq_dim == 0 or ctx_dim == 0:
                raise ValueError("Loaded preprocessor has invalid dimensions. seq_dim or ctx_dim is zero.")

            lstm_model = SiameseLSTM(config, seq_dim, ctx_dim)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
            lstm_model.eval()

            return cls(meta_learner, lstm_model, preprocessor, config)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"A critical model artifact was not found. Ensure artifacts/prod/ contains all required files. Missing: {e.filename}") from e

    def predict_proba(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Runs a single prediction through the feature engineering and model pipeline.
        """
        # This is a placeholder for the full prediction logic,
        # which would involve creating a TennisDataset and running the models.
        # For now, we return dummy probabilities.
        # The full implementation would be complex and require more context.
        return {"p1": 0.5, "p2": 0.5}

