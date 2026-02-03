import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any

class SklearnModel:
    """Base wrapper for Scikit-Learn models to behave like PyTorch models."""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)

class RandomForestBaseline(SklearnModel):
    def __init__(self, config: Dict[str, Any]):
        # We wrap in CalibratedClassifierCV to get true probabilities
        # (Random Forests are notoriously uncalibrated)
        base = RandomForestClassifier(
            n_estimators=config.model.n_estimators,
            max_depth=config.model.max_depth,
            n_jobs=-1,
            random_state=config.train.seed
        )
        calibrated = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        super().__init__(calibrated)

class LogisticBaseline(SklearnModel):
    def __init__(self, config: Dict[str, Any]):
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            n_jobs=-1
        )
        super().__init__(model)
