import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any
from src.config import ProjectConfig

class SklearnModel:
    def __init__(self, model: Any) -> None:
        self.model: Any = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)

class RandomForestBaseline(SklearnModel):
    def __init__(self, config: ProjectConfig) -> None:
        rf_params: Dict[str, Any] = config.models.random_forest.hyperparameters
        
        base: RandomForestClassifier = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", 8),
            min_samples_split=rf_params.get("min_samples_split", 2),
            n_jobs=-1,
            random_state=42
        )
        calibrated: CalibratedClassifierCV = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        super().__init__(calibrated)

class LogisticBaseline(SklearnModel):
    def __init__(self, config: ProjectConfig) -> None:
        lr_params: Dict[str, Any] = config.models.logistic_regression.hyperparameters
        
        model: LogisticRegression = LogisticRegression(
            C=lr_params.get("C", 1.0),
            solver='lbfgs',
            max_iter=lr_params.get("max_iter", 1000),
            n_jobs=-1
        )
        super().__init__(model)