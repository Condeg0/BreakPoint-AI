import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from src.models.nn import SiameseLSTM
from sklearn.metrics import roc_auc_score

class Tuner:
    def __init__(self, config, train_ds, val_ds):
        self.base_config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(self, trial):
        # 1. Define Search Space
        # We allow the model to be much smaller (16 hidden units) or larger
        hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        # 2. Setup DataLoaders (Dynamic Batch Size)
        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)

        # 3. Setup Model
        # Create a temp config to pass to the model
        temp_config = self.base_config.copy(deep=True)
        temp_config.model.hidden_size = hidden_size
        temp_config.model.num_layers = num_layers
        temp_config.model.dropout = dropout

        input_dim = self.train_ds.seq_matrix.shape[1]
        context_dim = self.train_ds.ctx_matrix.shape[1]

        model = SiameseLSTM(temp_config, input_dim, context_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # 4. Quick Training Loop (Pruning enabled)
        # We train for fewer epochs just to find the best potential
        for epoch in range(10):
            model.train()
            for seq_a, seq_b, ctx, y in train_loader:
                seq_a, seq_b, ctx, y = seq_a.to(self.device), seq_b.to(self.device), ctx.to(self.device), y.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                logits = model(seq_a, seq_b, ctx)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            # Validation
            auc = self._evaluate(model, val_loader)

            # Report to Optuna for pruning (stop bad trials early)
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return auc

    def _evaluate(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq_a, seq_b, ctx, y in loader:
                seq_a, seq_b, ctx = seq_a.to(self.device), seq_b.to(self.device), ctx.to(self.device)
                logits = model(seq_a, seq_b, ctx)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y.numpy())
        return roc_auc_score(all_labels, all_preds)

    def optimize(self, n_trials=20):
        print(f"\n>>> 🧠 OPTUNA: Starting Hyperparameter Search ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        print("\n>>> ✅ Best Trial found:")
        print(f"    Value (AUC): {study.best_value:.4f}")
        print("    Params: ")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")

        return study.best_params
