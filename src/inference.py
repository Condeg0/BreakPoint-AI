import joblib
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from src.models.stacking import StackingMetaLearner
from src.models.nn import SiameseLSTM
from src.models.baselines import RandomForestBaseline, LogisticBaseline
from src.data import Preprocessor, TennisDataset
from src.config import ProjectConfig
from src.logger import get_logger

logger = get_logger(__name__)

class MetaLearnerPipeline:
    def __init__(self, meta_learner: StackingMetaLearner, base_models: Dict[str, Any], preprocessor: Preprocessor, config: ProjectConfig):
        self.meta_learner = meta_learner
        self.base_models = base_models
        self.preprocessor = preprocessor
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "lstm" in self.base_models:
            self.base_models["lstm"].to(self.device)

    @classmethod
    def load_frozen_model(cls, base_path: Path):
        try:
            config = ProjectConfig.load(str(Path("configs/config.yaml")))
            
            preprocessor_path = base_path / "global_preprocessor.pkl"
            if not preprocessor_path.exists():
                preprocessor_path = base_path / "stacking" / "global_preprocessor.pkl"
            preprocessor = Preprocessor(config).load(preprocessor_path)

            stacker = StackingMetaLearner(config, base_path)
            try:
                stacker = stacker.load()
            except Exception as e:
                logger.warning(f"Standard load failed: {e}. Extracting raw artifact.")
                raw_data = joblib.load(base_path / "stacking" / "meta_learner.joblib")
                if isinstance(raw_data, dict):
                    stacker.meta_model = raw_data.get('model')
                    stacker.model_names = raw_data.get('features', [])
                else:
                    stacker.meta_model = raw_data
                    stacker.model_names = []

            if not stacker.model_names:
                available_models = [p.name for p in base_path.iterdir() if p.is_dir() and p.name in ["lstm", "random_forest", "logistic_regression", "xgboost"]]
                logger.error(f"Missing model_names. Dynamically assigning: {available_models}")
                stacker.model_names = sorted(available_models)

            base_models = {}
            for base_name in stacker.model_names:
                model_dir = base_path / base_name
                
                if base_name == "lstm":
                    seq_dim = len(preprocessor.seq_indices)
                    ctx_dim = len(preprocessor.ctx_indices)
                    lstm_model = SiameseLSTM(config, seq_dim, ctx_dim)
                    lstm_model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=torch.device('cpu')))
                    lstm_model.eval()
                    base_models["lstm"] = lstm_model
                elif base_name == "random_forest":
                    rf = RandomForestBaseline(config)
                    rf.load(model_dir / "model.joblib")
                    base_models["random_forest"] = rf
                elif base_name == "logistic_regression":
                    lr = LogisticBaseline(config)
                    lr.load(model_dir / "model.joblib")
                    base_models["logistic_regression"] = lr
                # FIXED: Missing XGBoost logic injected
                elif base_name == "xgboost":
                    from src.models.xgb import XGBoostModel
                    xgb = XGBoostModel(config)
                    xgb.load(model_dir / "model.joblib")
                    base_models["xgboost"] = xgb

            return cls(stacker, base_models, preprocessor, config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {str(e)}") from e

    def predict_batch(self, combined_df: pd.DataFrame) -> List[Dict[str, Any]]:
        try:
            if 'tourney_date' in combined_df.columns:
                combined_df['tourney_date'] = pd.to_datetime(combined_df['tourney_date'].astype(str), format="%Y%m%d", errors='coerce')

            inference_df = combined_df[combined_df['is_inference'] == True].copy()
            if inference_df.empty:
                return []

            seq_len = self.config.architecture.seq_len if hasattr(self.config, 'architecture') else 10
            base_preds = {}

            # 1. Execute LSTM Sequence Tensors
            if "lstm" in self.base_models:
                ds_lstm = TennisDataset(combined_df, self.preprocessor, mode="lstm", seq_len=seq_len)
                inference_indices = combined_df.index[combined_df['is_inference'] == True].tolist()
                
                lstm_probs = []
                with torch.no_grad():
                    for idx in inference_indices:
                        seq_a, seq_b, ctx, _ = ds_lstm[idx]
                        seq_a = seq_a.unsqueeze(0).to(self.device)
                        seq_b = seq_b.unsqueeze(0).to(self.device)
                        ctx = ctx.unsqueeze(0).to(self.device)
                        
                        logits = self.base_models["lstm"](seq_a, seq_b, ctx)
                        prob = torch.sigmoid(logits).cpu().numpy().flatten()[0]
                        lstm_probs.append(prob)
                
                base_preds["lstm"] = np.array(lstm_probs)

            # 2. Execute Tabular Baselines
            tabular_models = [m for m in self.meta_learner.model_names if m != "lstm" and m in self.base_models]
            if tabular_models:
                ds_tab = TennisDataset(combined_df, self.preprocessor, mode="tabular")
                ctx_matrix_full = ds_tab.ctx_matrix
                ctx_matrix_inf = ctx_matrix_full[combined_df['is_inference'] == True]
                
                for m_name in tabular_models:
                    model = self.base_models[m_name]
                    prob = model.predict_proba(ctx_matrix_inf)
                    base_preds[m_name] = prob[:, 1] if isinstance(prob, np.ndarray) and prob.ndim == 2 else prob

            # 3. Fault-Tolerant Meta-Learner Fusion
            valid_preds = []
            for name in self.meta_learner.model_names:
                if name in base_preds:
                    valid_preds.append(base_preds[name])
                else:
                    logger.error(f"Missing predictions for {name}. Filling with neutral EV (0.5).")
                    valid_preds.append(np.full(len(inference_df), 0.5))
                    
            X_meta = np.column_stack(valid_preds)
            
            try:
                final_probs = self.meta_learner.meta_model.predict_proba(X_meta)[:, 1]
            except Exception:
                final_probs = np.mean(X_meta, axis=1)

            # 4. Format Output
            results = []
            for i, row in enumerate(inference_df.to_dict(orient="records")):
                p1_win = float(final_probs[i])
                
                # Intercept the standardized column names from FeatureEngineer
                p1_name = row.get("player", row.get("winner_name", "Unknown"))
                p2_name = row.get("opponent", row.get("loser_name", "Unknown"))
                
                results.append({
                    "player_1": p1_name,
                    "player_2": p2_name,
                    "player_1_win_probability": round(p1_win, 4),
                    "player_2_win_probability": round(1.0 - p1_win, 4),
                    "confidence_spread": round(abs(p1_win - (1.0 - p1_win)), 4)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch Execution Error: {e}")
            raise e