import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from typing import Dict, List, Any, Self, Union

from src.config import ProjectConfig
from src.logger import get_logger

logger = get_logger(__name__)

class StackingMetaLearner:
    def __init__(self, config: ProjectConfig, artifact_dir: Path) -> None:
        self.config: ProjectConfig = config
        self.artifact_dir: Path = artifact_dir
        self.stacking_dir: Path = self.artifact_dir / "stacking"
        self.stacking_dir.mkdir(parents=True, exist_ok=True)
        
        meta_learner_type: str = config.models.stacking.meta_learner
        self.meta_model: Union[LogisticRegression, XGBClassifier]
        if meta_learner_type == "logistic_regression":
            self.meta_model = LogisticRegression(C=10.0, solver='lbfgs')
        elif meta_learner_type == "xgboost":
            self.meta_model = XGBClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                subsample=0.8, objective="binary:logistic", eval_metric="auc",
                n_jobs=-1, random_state=42
            )
        else:
            raise ValueError(f"Unsupported meta-learner: {meta_learner_type}")
            
        self.model_names: List[str] = []

    def fit(self, val_predictions: Dict[str, np.ndarray], y_val: np.ndarray) -> None:
        """
        val_predictions: dict mapping model_name -> 1D numpy array of validation probabilities
        """
        self.model_names = sorted(list(val_predictions.keys()))
        X_meta: np.ndarray = np.column_stack([val_predictions[name] for name in self.model_names])
        
        self.meta_model.fit(X_meta, y_val)
        
        logger.info("\n>>> Meta-Learner Weights Learned:")
        if hasattr(self.meta_model, 'coef_'):
            for name, weight in zip(self.model_names, self.meta_model.coef_[0]):
                logger.info(f"    - {name}: {weight:.4f}")
            logger.info(f"    - Intercept: {self.meta_model.intercept_[0]:.4f}")
        else:
            logger.info("    (XGBoost meta-learner — weights not expressed as linear coefficients)")

    def predict_proba(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Guarantee we pull probabilities in the exact same order they were fitted
        X_meta: np.ndarray = np.column_stack([predictions_dict[name] for name in self.model_names])
        return self.meta_model.predict_proba(X_meta)[:, 1]

    def save(self) -> None:
        save_dict: Dict[str, Any] = {
            'model': self.meta_model,
            'features': self.model_names
        }
        joblib.dump(save_dict, self.stacking_dir / "meta_learner.joblib")
        logger.info(f"Stacking Meta-Learner saved to {self.stacking_dir}")

    def load(self) -> Self:
        load_dict: Dict[str, Any] = joblib.load(self.stacking_dir / "meta_learner.joblib")
        self.meta_model = load_dict['model']
        self.model_names = load_dict['features']
        logger.info(f"Stacking Meta-Learner loaded from {self.stacking_dir}")
        return self