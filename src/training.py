import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Type, Union, List, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from src.logger import get_logger
from src.config import ProjectConfig
from src.data import TennisDataset
from src.models.nn import SiameseLSTM
from src.models.baselines import RandomForestBaseline, LogisticBaseline, SklearnModel
from src.models.xgb import XGBoostModel

logger = get_logger(__name__)

class Trainer:
    def __init__(self, config: ProjectConfig, run_dir: Path) -> None:
        self.config: ProjectConfig = config
        self.run_dir: Path = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_ds: TennisDataset, val_ds: TennisDataset, model_name: str) -> Union[SklearnModel, XGBoostModel, SiameseLSTM]:
        logger.info("\n" + "="*50)
        logger.info(f">>> 🔍 INITIALIZING TRAINING PIPELINE: {model_name.upper()}")
        logger.info("="*50)

        if model_name == "lstm":
            return self._train_lstm(train_ds, val_ds)
        elif model_name == "random_forest":
            return self._train_sklearn(train_ds, val_ds, RandomForestBaseline)
        elif model_name == "logistic_regression":
            return self._train_sklearn(train_ds, val_ds, LogisticBaseline)
        elif model_name == "xgboost":
            return self._train_xgb(train_ds, val_ds)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _train_sklearn(self, train_ds: TennisDataset, val_ds: TennisDataset, model_cls: Type[SklearnModel]) -> SklearnModel:
        logger.info(f"Training {model_cls.__name__}...")
        model: SklearnModel = model_cls(self.config)

        X_train: np.ndarray = train_ds.ctx_matrix
        y_train: np.ndarray = train_ds.y_vector
        X_val: np.ndarray = val_ds.ctx_matrix
        y_val: np.ndarray = val_ds.y_vector
        
        model.fit(X_train, y_train)

        train_probs: np.ndarray = model.predict_proba(X_train)
        val_probs: np.ndarray = model.predict_proba(X_val)

        train_auc: float = roc_auc_score(y_train, train_probs)
        val_auc: float = roc_auc_score(y_val, val_probs)

        logger.info(f"[{model_cls.__name__}] Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        save_path: Path = self.run_dir / "model.joblib"
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        return model

    def _train_xgb(self, train_ds: TennisDataset, val_ds: TennisDataset) -> XGBoostModel:
        logger.info(f"Training XGBoostModel...")
        model: XGBoostModel = XGBoostModel(self.config)

        X_train: np.ndarray = train_ds.ctx_matrix
        y_train: np.ndarray = train_ds.y_vector
        X_val: np.ndarray = val_ds.ctx_matrix
        y_val: np.ndarray = val_ds.y_vector
        
        model.fit(X_train, y_train, X_val, y_val)

        train_probs: np.ndarray = model.predict_proba(X_train)
        val_probs: np.ndarray = model.predict_proba(X_val)

        train_auc: float = roc_auc_score(y_train, train_probs)
        val_auc: float = roc_auc_score(y_val, val_probs)

        best_iter: int = model.model.best_iteration
        logger.info(f"[XGBoost] Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Best Iteration: {best_iter}")

        save_path: Path = self.run_dir / "model.joblib"
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        return model

    def _train_lstm(self, train_ds: TennisDataset, val_ds: TennisDataset) -> SiameseLSTM:
        lstm_train_cfg = self.config.models.lstm.training
        
        train_loader: DataLoader = DataLoader(train_ds, batch_size=lstm_train_cfg.batch_size, shuffle=True)
        val_loader: DataLoader = DataLoader(val_ds, batch_size=lstm_train_cfg.batch_size, shuffle=False)

        input_dim: int = train_ds.seq_matrix.shape[1]
        context_dim: int = train_ds.ctx_matrix.shape[1]

        model: SiameseLSTM = SiameseLSTM(self.config, input_dim, context_dim).to(self.device)
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=lstm_train_cfg.learning_rate)
        criterion: nn.Module = nn.BCEWithLogitsLoss()

        best_val_auc: float = 0.0
        patience_counter: int = 0
        patience_limit: int = 5

        for epoch in range(lstm_train_cfg.epochs):
            model.train()
            train_loss: float = 0.0

            for seq_a, seq_b, ctx, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                seq_a: torch.Tensor = seq_a.to(self.device)
                seq_b: torch.Tensor = seq_b.to(self.device)
                ctx: torch.Tensor = ctx.to(self.device)
                y_tensor: torch.Tensor = y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits: torch.Tensor = model(seq_a, seq_b, ctx)
                loss: torch.Tensor = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss: float = train_loss / len(train_loader)
            val_auc, val_loss = self._evaluate_lstm(model, val_loader, criterion)

            logger.info(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), self.run_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logger.warning("Early stopping triggered.")
                    break

        logger.info(f"Best Val AUC: {best_val_auc:.4f}")
        # Load the best model state before returning
        model.load_state_dict(torch.load(self.run_dir / "best_model.pt"))
        return model

    def _evaluate_lstm(self, model: SiameseLSTM, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        total_loss: float = 0.0

        with torch.no_grad():
            for seq_a, seq_b, ctx, y in loader:
                seq_a: torch.Tensor = seq_a.to(self.device)
                seq_b: torch.Tensor = seq_b.to(self.device)
                ctx: torch.Tensor = ctx.to(self.device)
                y_target: torch.Tensor = y.to(self.device).unsqueeze(1)

                logits: torch.Tensor = model(seq_a, seq_b, ctx)
                loss: torch.Tensor = criterion(logits, y_target)

                total_loss += loss.item()
                probs: np.ndarray = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y.numpy())

        avg_loss: float = total_loss / len(loader)
        auc: float = roc_auc_score(all_labels, all_preds)
        return auc, avg_loss