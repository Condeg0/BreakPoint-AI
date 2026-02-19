import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.config import ProjectConfig
from src.features import FeatureEngineer
from src.data import load_and_split, TennisDataset, Preprocessor
from src.training import Trainer

def main():
    parser = argparse.ArgumentParser(description="BreakPoint AI: Orchestrator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from {config_path}...")
    config = ProjectConfig.load(config_path)

    print("\n>>> Phase 1: Data Ingestion")
    train_raw, val_raw, test_raw = load_and_split(config)

    print("\n>>> Phase 2: Feature Engineering")
    full_df = pd.concat([train_raw, val_raw, test_raw], axis=0).sort_values("tourney_date")
    engineer = FeatureEngineer(rolling_window=10)
    full_feat_df = engineer.generate_features(full_df)

    train_cutoff = pd.to_datetime(config.data.temporal_splits.train_cutoff)
    test_start = pd.to_datetime(config.data.temporal_splits.test_start)

    train_df = full_feat_df[full_feat_df['tourney_date'] <= train_cutoff]
    val_df = full_feat_df[(full_feat_df['tourney_date'] > train_cutoff) & (full_feat_df['tourney_date'] < test_start)]
    test_df = full_feat_df[full_feat_df['tourney_date'] >= test_start]

    print(f"Featured Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print("\n>>> Phase 3: Preprocessing Setup")
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_artifact_dir = Path(config.data.paths.artifact_dir) / run_id
    base_artifact_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = Preprocessor(config)
    preprocessor.fit(train_df)
    preprocessor.save(base_artifact_dir / "global_preprocessor.pkl")

    # Initialize Datasets for Train, Val, and TEST
    train_ds_tab = TennisDataset(train_df, preprocessor, mode="tabular")
    val_ds_tab   = TennisDataset(val_df, preprocessor, mode="tabular")
    test_ds_tab  = TennisDataset(test_df, preprocessor, mode="tabular")

    lstm_seq_len = config.models.lstm.architecture.seq_len
    train_ds_lstm = TennisDataset(train_df, preprocessor, mode="lstm", seq_len=lstm_seq_len)
    val_ds_lstm   = TennisDataset(val_df, preprocessor, mode="lstm", seq_len=lstm_seq_len)
    test_ds_lstm  = TennisDataset(test_df, preprocessor, mode="lstm", seq_len=lstm_seq_len)

    print("\n>>> Phase 4: Model Training Execution")
    for model_name in config.pipeline.models_to_train:
        print(f"\n--- Initiating Run for: {model_name} ---")
        
        model_artifact_dir = base_artifact_dir / model_name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(config, model_artifact_dir)

        # 1. Train
        if model_name == "lstm":
            model = trainer.train(train_ds_lstm, val_ds_lstm, model_name=model_name)
        else:
            if model_name == "xgboost":
                model = trainer.train(train_ds_tab, val_ds_tab, model_name=model_name)
            model = trainer.train(train_ds_tab, val_ds_tab, model_name=model_name)

        # 2. Log Hyperparameters
        params_to_save = {}
        if model_name == "lstm":
            params_to_save["architecture"] = config.models.lstm.architecture.model_dump()
            params_to_save["training"] = config.models.lstm.training.model_dump()
        else:
            model_cfg = getattr(config.models, model_name)
            params_to_save = model_cfg.hyperparameters

        with open(model_artifact_dir / "hyperparameters.json", "w") as f:
            json.dump(params_to_save, f, indent=4)

        # 3. Evaluation on Test Set
        if config.pipeline.run_evaluation:
            print(f">>> Phase 5: Evaluation & Artifact Logging ({model_name})")
            from src.evaluation import Evaluator
            evaluator = Evaluator(model_artifact_dir)

            if model_name == "lstm":
                model.eval()
                all_probs, all_targets = [], []
                device = next(model.parameters()).device
                loader = torch.utils.data.DataLoader(test_ds_lstm, batch_size=256, shuffle=False)

                with torch.no_grad():
                    for seq_a, seq_b, ctx, y in loader:
                        seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                        logits = model(seq_a, seq_b, ctx)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_probs.extend(probs)
                        all_targets.extend(y.numpy())

                y_prob, y_true = np.array(all_probs), np.array(all_targets)

                # SHAP preparation for LSTM
                indices = np.random.choice(len(train_ds_lstm), 50, replace=False)
                X_train_shap = np.array([
                    np.concatenate([train_ds_lstm[i][0].numpy().flatten(), 
                                    train_ds_lstm[i][1].numpy().flatten(), 
                                    train_ds_lstm[i][2].numpy().flatten()])
                    for i in indices
                ])
                
                seq_feats = config.data.features.sequence
                ctx_feats = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]
                all_names = [f"P1_{f}_t-{lstm_seq_len - i}" for i in range(lstm_seq_len) for f in seq_feats] + \
                            [f"P2_{f}_t-{lstm_seq_len - i}" for i in range(lstm_seq_len) for f in seq_feats] + \
                            ctx_feats

                evaluator.generate_report(y_true, y_prob, model=model, X_train=X_train_shap, model_name="lstm", feature_names=all_names)

            else:
                X_test = test_ds_tab.ctx_matrix
                y_true = test_ds_tab.y_vector
                y_prob = model.predict_proba(X_test)

                ctx_feature_names = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]
                indices = np.random.choice(len(train_ds_tab), min(500, len(train_ds_tab)), replace=False)
                X_train_df = pd.DataFrame(train_ds_tab.ctx_matrix[indices], columns=ctx_feature_names)

                evaluator.generate_report(y_true, y_prob, model=model, X_train=X_train_df, model_name=model_name, feature_names=ctx_feature_names)

    print(f"\n>>> Global Execution Complete. All artifacts saved to: {base_artifact_dir}")

if __name__ == "__main__":
    main()