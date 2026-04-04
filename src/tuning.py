import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

from src.config import ProjectConfig
from src.data import TennisDataset
from src.models.nn import SiameseLSTM
from sklearn.metrics import roc_auc_score

class Tuner:
    def __init__(self, config: ProjectConfig, train_ds: TennisDataset, val_ds: TennisDataset) -> None:
        self.base_config: ProjectConfig = config
        self.train_ds: TennisDataset = train_ds
        self.val_ds: TennisDataset = val_ds
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Define Search Space
        hidden_size: int = trial.suggest_int("hidden_size", 32, 128, step=16)
        num_layers: int = trial.suggest_int("num_layers", 1, 3)
        dropout: float = trial.suggest_float("dropout", 0.2, 0.5)
        lr: float = trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True)
        weight_decay: float = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        fusion_dim: int = trial.suggest_categorical("fusion_dim", [64, 128, 256])
        batch_size: int = trial.suggest_categorical("batch_size", [64, 128, 256])

        # 2. Setup DataLoaders
        train_loader: DataLoader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader: DataLoader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)

        # 3. Setup Model
        temp_config: ProjectConfig = self.base_config.copy(deep=True)
        temp_config.models.lstm.architecture.hidden_size = hidden_size
        temp_config.models.lstm.architecture.num_layers = num_layers
        temp_config.models.lstm.architecture.dropout = dropout
        temp_config.models.lstm.architecture.fusion_dim = fusion_dim

        input_dim: int = self.train_ds.seq_matrix.shape[1]
        context_dim: int = self.train_ds.ctx_matrix.shape[1]

        model: SiameseLSTM = SiameseLSTM(temp_config, input_dim, context_dim).to(self.device)
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion: nn.Module = nn.BCEWithLogitsLoss()

        # 4. Quick Training Loop (15 epochs for better convergence signal at low LR)
        for epoch in range(15):
            model.train()
            for seq_a, seq_b, ctx, y in train_loader:
                seq_a, seq_b, ctx, y = seq_a.to(self.device), seq_b.to(self.device), ctx.to(self.device), y.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                logits: torch.Tensor = model(seq_a, seq_b, ctx)
                loss: torch.Tensor = criterion(logits, y)
                loss.backward()
                optimizer.step()

            auc: float = self._evaluate(model, val_loader)
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return auc

    def _evaluate(self, model: SiameseLSTM, loader: DataLoader) -> float:
        model.eval()
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for seq_a, seq_b, ctx, y in loader:
                seq_a, seq_b, ctx = seq_a.to(self.device), seq_b.to(self.device), ctx.to(self.device)
                logits: torch.Tensor = model(seq_a, seq_b, ctx)
                probs: np.ndarray = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y.numpy())
        return roc_auc_score(all_labels, all_preds)

    def optimize(self, n_trials: int = 20) -> Dict[str, Any]:
        print(f"\n>>> 🧠 OPTUNA: Starting Hyperparameter Search ({n_trials} trials)...")
        study: optuna.Study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        print("\n>>> ✅ Best Trial found:")
        print(f"    Value (AUC): {study.best_value:.4f}")
        print("    Params: ")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")

        return study.best_params
