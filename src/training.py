import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from src.models.nn import SiameseLSTM
from src.models.baselines import RandomForestBaseline, LogisticBaseline

class Trainer:
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

    def train(self, train_ds, val_ds):
        model_name = self.config.model.name

        # --- NEW CODE: FEATURE VERIFICATION ---
        print("\n" + "="*50)
        print(f">>> 🔍 FINAL FEATURE CHECK BEFORE TRAINING ({model_name.upper()})")
        print("="*50)
        
        preproc = train_ds.preprocessor
        
        # 1. Get Context Features (Used by Baselines AND LSTM)
        ctx_names = [preproc.feature_names[i] for i in preproc.ctx_indices]
        print(f"\n[1] CONTEXT FEATURES (Static Input) - Count: {len(ctx_names)}")
        print(f"    Used by: RF, LogReg, and LSTM (Fusion Layer)")
        print(f"    List: {ctx_names}")

        # 2. Get Sequence Features (Used by LSTM ONLY)
        if model_name == "lstm":
            seq_names = [preproc.feature_names[i] for i in preproc.seq_indices]
            print(f"\n[2] SEQUENCE FEATURES (History Input) - Count: {len(seq_names)}")
            print(f"    Used by: LSTM (Recurrent Layer History)")
            print(f"    List: {seq_names}")
            
        print("\n" + "="*60 + "\n")

        if model_name == "lstm":
            # Just grab one item manually
            seq_a, seq_b, feats, y = train_ds[0]
            print(f"  Target: {y}")

        if model_name == "lstm":
            return self._train_lstm(train_ds, val_ds)
        elif model_name == "rf":
            return self._train_sklearn(train_ds, val_ds, RandomForestBaseline)
        elif model_name == "logreg":
            return self._train_sklearn(train_ds, val_ds, LogisticBaseline)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _train_sklearn(self, train_ds, val_ds, model_cls):
        print(f"Training {model_cls.__name__}...")
        model = model_cls(self.config)

        # Baselines only use the Context Matrix (Rolling Stats + Metadata)
        X_train, y_train = train_ds.ctx_matrix, train_ds.y_vector
        X_val, y_val = val_ds.ctx_matrix, val_ds.y_vector
        
        model.fit(X_train, y_train)

        train_probs = model.predict_proba(X_train)
        val_probs = model.predict_proba(X_val)

        train_auc = roc_auc_score(y_train, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        save_path = self.run_dir / "model.joblib"
        model.save(save_path)
        print(f"Model saved to {save_path}")
        return model

    def _train_lstm(self, train_ds, val_ds):
        train_loader = DataLoader(train_ds, batch_size=self.config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.train.batch_size, shuffle=False)

        # --- FIX: Correctly grab dimensions from the new Dataset structure ---
        # seq_matrix holds the raw history stats (input_size for LSTM)
        # ctx_matrix holds the static features (input_size for Fusion Layer)
        input_dim = train_ds.seq_matrix.shape[1]
        context_dim = train_ds.ctx_matrix.shape[1]

        print(f"Initializing LSTM with Input Dim={input_dim}, Context Dim={context_dim}")

        model = SiameseLSTM(self.config, input_dim, context_dim).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.train.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0
        patience_counter = 0
        patience_limit = 5

        print(f"Starting LSTM Training ({self.config.train.epochs} epochs)...")

        for epoch in range(self.config.train.epochs):
            model.train()
            train_loss = 0.0

            for seq_a, seq_b, ctx, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                seq_a = seq_a.to(self.device)
                seq_b = seq_b.to(self.device)
                ctx = ctx.to(self.device)
                y = y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = model(seq_a, seq_b, ctx)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            val_auc, val_loss = self._evaluate_lstm(model, val_loader, criterion)

            print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), self.run_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print("Early stopping triggered.")
                    break

        print(f"Best Val AUC: {best_val_auc:.4f}")
        return model

    def _evaluate_lstm(self, model, loader, criterion):
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for seq_a, seq_b, ctx, y in loader:
                seq_a = seq_a.to(self.device)
                seq_b = seq_b.to(self.device)
                ctx = ctx.to(self.device)
                y_target = y.to(self.device).unsqueeze(1)

                logits = model(seq_a, seq_b, ctx)
                loss = criterion(logits, y_target)

                total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y.numpy())

        avg_loss = total_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        return auc, avg_loss
