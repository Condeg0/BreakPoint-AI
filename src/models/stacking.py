import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

class StackingMetaLearner:
    def __init__(self, config, artifact_dir: Path):
        self.config = config
        self.artifact_dir = artifact_dir
        self.stacking_dir = self.artifact_dir / "stacking"
        self.stacking_dir.mkdir(parents=True, exist_ok=True)
        
        meta_learner_type = config.models.stacking.meta_learner
        if meta_learner_type == "logistic_regression":
            # Strict L2 regularization to prevent meta-overfitting
            #self.meta_model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs')
            self.meta_model = LogisticRegression(C=10.0, solver='lbfgs')
        else:
            raise ValueError(f"Unsupported meta-learner: {meta_learner_type}")
            
        self.model_names = []

    def fit(self, val_predictions: dict, y_val: np.ndarray):
        """
        val_predictions: dict mapping model_name -> 1D numpy array of validation probabilities
        """
        # Enforce deterministic ordering of features
        self.model_names = sorted(list(val_predictions.keys()))
        X_meta = np.column_stack([val_predictions[name] for name in self.model_names])
        
        self.meta_model.fit(X_meta, y_val)
        
        print("\n>>> Meta-Learner Weights Learned:")
        for name, weight in zip(self.model_names, self.meta_model.coef_[0]):
            print(f"    - {name}: {weight:.4f}")
        print(f"    - Intercept: {self.meta_model.intercept_[0]:.4f}")

    def predict_proba(self, predictions_dict: dict) -> np.ndarray:
        # Guarantee we pull probabilities in the exact same order they were fitted
        X_meta = np.column_stack([predictions_dict[name] for name in self.model_names])
        return self.meta_model.predict_proba(X_meta)[:, 1]

    def save(self):
        save_dict = {
            'model': self.meta_model,
            'features': self.model_names
        }
        joblib.dump(save_dict, self.stacking_dir / "meta_learner.joblib")
        print(f"Stacking Meta-Learner saved to {self.stacking_dir}")

    def load(self):
        load_dict = joblib.load(self.stacking_dir / "meta_learner.joblib")
        self.meta_model = load_dict['model']
        self.model_names = load_dict['features']
        return self