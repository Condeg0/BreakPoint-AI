import xgboost as xgb
import joblib
from pathlib import Path

class XGBoostModel:
    def __init__(self, config):
        xgb_params = config.models.xgboost.hyperparameters
        
        self.model = xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 200),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=20,
            n_jobs=-1,
            random_state=42
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)