import argparse
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
import numpy as np

# Local imports
from src.config import ProjectConfig
from src.features import FeatureEngineer
from src.data import load_and_split, TennisDataset, Preprocessor
from src.training import Trainer

def main():
    parser = argparse.ArgumentParser(description="BreakPoint AI: Production Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from {config_path}...")
    config = ProjectConfig.load(config_path)

    # Setup Artifact Directory (Where output and model information will be saved)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / config.model.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Ingestion & Splitting
    print(">>> Phase 1: Data Ingestion")
    train_raw, val_raw, test_raw = load_and_split(config)

    # 3. Feature Engineering
    print(">>> Phase 2: Feature Engineering")
    full_df = pd.concat([train_raw, val_raw, test_raw], axis=0).sort_values("tourney_date")

    engineer = FeatureEngineer(rolling_window=10)
    full_feat_df = engineer.generate_features(full_df)

    # Apply date split
    train_df = full_feat_df[full_feat_df['tourney_date'] <= config.data.train_cutoff]
    val_df = full_feat_df[(full_feat_df['tourney_date'] > config.data.train_cutoff) &
                          (full_feat_df['tourney_date'] < config.data.test_start)]
    test_df = full_feat_df[full_feat_df['tourney_date'] >= config.data.test_start]

    print(f"Featured Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # --- DEBUG: SAVE PROCESSED DATASETS ---
    debug_dir = Path("data/processed")
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> 💾 DEBUG: Saving intermediate datasets to {debug_dir}...")
    train_df.to_csv(debug_dir / "train_engineered.csv", index=False)
    val_df.to_csv(debug_dir / "val_engineered.csv", index=False)
    test_df.to_csv(debug_dir / "test_engineered.csv", index=False)
    print("   - train_engineered.csv")
    print("   - val_engineered.csv")
    print("   - test_engineered.csv")

    # 4. Preprocessing
    print("\n>>> Phase 3: Preprocessing (Fit on Train ONLY)")
    preprocessor = Preprocessor(config)
    preprocessor.fit(train_df)
    preprocessor.save(run_dir / "preprocessor.pkl")

    # 5. Create Datasets
    dataset_mode = "lstm" if config.model.name == "lstm" else "tabular"
    train_ds = TennisDataset(train_df, preprocessor, mode=dataset_mode)
    val_ds   = TennisDataset(val_df, preprocessor, mode=dataset_mode)
    test_ds  = TennisDataset(test_df, preprocessor, mode=dataset_mode)

    # 6. Tuning & Training
    if config.train.tuning and config.model.name == "lstm":
        from src.tuning import Tuner
        tuner = Tuner(config, train_ds, val_ds)
        best_params = tuner.optimize(n_trials=20)

        # Apply best params to the config
        print("\n>>> 🚀 Re-configuring model with optimized parameters...")
        config.model.hidden_size = best_params["hidden_size"]
        config.model.num_layers = best_params["num_layers"]
        config.model.dropout = best_params["dropout"]
        config.train.learning_rate = best_params["learning_rate"]
        config.train.batch_size = best_params["batch_size"]

        # Save best params for reference
        with open(run_dir / "best_params.yaml", "w") as f:
            yaml.dump(best_params, f)

    print(f"\n>>> Phase 4: Training Model ({config.model.name})")
    trainer = Trainer(config, run_dir)
    model = trainer.train(train_ds, val_ds)

    # 7. Final Evaluation on TEST SET
    print(f"\n>>> Phase 5: Final Evaluation on Test Set ({config.data.test_start}+)")

    try:
        from src.evaluation import Evaluator
        evaluator = Evaluator(run_dir)

        if config.model.name == "lstm":
            model.eval()
            all_probs = []
            all_targets = []
            device = next(model.parameters()).device
            loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

            with torch.no_grad():
                for seq_a, seq_b, ctx, y in loader:
                    seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                    logits = model(seq_a, seq_b, ctx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    all_targets.extend(y.numpy())

            y_prob = np.array(all_probs)
            y_true = np.array(all_targets)

            # --- PREPARE DATA FOR SHAP (FLATTEN) ---
            print("   - Preparing data for LSTM SHAP...")
            background_samples = []

            # Using random indices to sample from Dataset
            indices = np.random.choice(len(train_ds), 50, replace=False)

            for idx in indices:
                s_a, s_b, ctx, _ = train_ds[idx] # these are tensors
                # Flatten everything into one long 1D array
                flat = np.concatenate([
                    s_a.numpy().flatten(),
                    s_b.numpy().flatten(),
                    ctx.numpy().flatten()
                ])
                background_samples.append(flat)

            X_train_shap = np.array(background_samples)

            # Create proper feature names for the flattened columns
            seq_feats = config.data.sequence_features
            ctx_feats = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]

            all_names = []
            # Add names for Player A Sequence
            for i in range(config.model.seq_len):
                for f in seq_feats: all_names.append(f"P1_{f}_t-{config.model.seq_len - i}")
            # Add names for Player B Sequence
            for i in range(config.model.seq_len):
                for f in seq_feats: all_names.append(f"P2_{f}_t-{config.model.seq_len - i}")
            # Add Context names
            all_names.extend(ctx_feats)

            evaluator.generate_report(y_true, y_prob, model=model, X_train=X_train_shap, model_name="lstm", feature_names=all_names)

        else:
            # Baselines (RF / LogReg)
            X_test = test_ds.ctx_matrix
            y_true = test_ds.y_vector
            y_prob = model.predict_proba(X_test)

            # Get Context Feature Names
            ctx_feature_names = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]

            # Sample training data for SHAP background
            indices = np.random.choice(len(train_ds), 500, replace=False)
            X_train_sample = train_ds.ctx_matrix[indices]
            X_train_df = pd.DataFrame(X_train_sample, columns=ctx_feature_names)

            evaluator.generate_report(
                y_true,
                y_prob,
                model=model,
                X_train=X_train_df,
                model_name=config.model.name,
                feature_names=ctx_feature_names
            )


        # Print Text Metrics to Console
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, (y_prob > 0.5).astype(int))
        print(f"FINAL TEST RESULTS: AUC={auc:.4f}, Accuracy={acc:.4f}")

    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n>>> Pipeline Complete. Artifacts: {run_dir}")

if __name__ == "__main__":
    main()
