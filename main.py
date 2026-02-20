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
    
    # Load frozen preprocessor if only running stacking
    if config.pipeline.models_to_train == ["stacking"]:
        print("    > Stacking Mode: Loading frozen preprocessor to guarantee feature parity.")
        source_dir = Path(config.pipeline.stacking_base_artifact_dir)
        preprocessor.load(source_dir / "global_preprocessor.pkl")
        preprocessor.save(base_artifact_dir / "global_preprocessor.pkl")
    else:
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

    # ---------------------------------------------------------
    # Phase 4: BASE MODEL TRAINING
    # ---------------------------------------------------------
    print("\n>>> Phase 4: Model Execution")
    trained_models = {}

    base_models = [m for m in config.pipeline.models_to_train if m != "stacking"]
    
    for model_name in base_models:
        print(f"\n--- Initiating Base Model Run: {model_name} ---")
        
        model_artifact_dir = base_artifact_dir / model_name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(config, model_artifact_dir)

        if model_name == "lstm":
            model = trainer.train(train_ds_lstm, val_ds_lstm, model_name=model_name)
        else:
            model = trainer.train(train_ds_tab, val_ds_tab, model_name=model_name)
            
        trained_models[model_name] = model

        params_to_save = {}
        if model_name == "lstm":
            params_to_save["architecture"] = config.models.lstm.architecture.model_dump()
            params_to_save["training"] = config.models.lstm.training.model_dump()
        else:
            model_cfg = getattr(config.models, model_name)
            params_to_save = model_cfg.hyperparameters

        with open(model_artifact_dir / "hyperparameters.json", "w") as f:
            json.dump(params_to_save, f, indent=4)

    # ---------------------------------------------------------
    # Phase 4.5: DECOUPLED STACKING ARCHITECTURE
    # ---------------------------------------------------------
    if "stacking" in config.pipeline.models_to_train:
        print("\n" + "="*50)
        print(">>>  INITIATING STACKING META-LEARNER (FROM ARTIFACTS)")
        print("="*50)
        
        if not config.pipeline.stacking_base_artifact_dir:
            raise ValueError("stacking_base_artifact_dir must be defined in config.yaml to train the Stacker.")
            
        source_dir = Path(config.pipeline.stacking_base_artifact_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Base artifact directory not found: {source_dir}")

        val_preds_dict = {}
        expected_bases = ["lstm", "xgboost", "random_forest", "logistic_regression"]
        found_bases = [m for m in expected_bases if (source_dir / m).exists()]
        
        if not found_bases:
            raise ValueError(f"No pre-trained base models found in {source_dir}")
            
        print(f"Loading pre-trained base models for Meta-Feature generation: {found_bases}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for base_name in found_bases:
            model_dir = source_dir / base_name
            print(f"    > Generating Validation OOF predictions for {base_name}...")
            
            if base_name == "lstm":
                from src.models.nn import SiameseLSTM  # <-- Add this missing import
                
                with open(model_dir / "hyperparameters.json", "r") as f:
                    seq_len = json.load(f)["architecture"]["seq_len"]
                
                # Initialize a localized LSTM model and load weights
                model = SiameseLSTM(config, train_ds_lstm.seq_matrix.shape[1], train_ds_lstm.ctx_matrix.shape[1]).to(device)
                model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
                model.eval()
                
                val_probs = []
                loader = torch.utils.data.DataLoader(val_ds_lstm, batch_size=256, shuffle=False)
                with torch.no_grad():
                    for seq_a, seq_b, ctx, _ in loader:
                        seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                        probs = torch.sigmoid(model(seq_a, seq_b, ctx)).cpu().numpy().flatten()
                        val_probs.extend(probs)
                val_preds_dict[base_name] = np.array(val_probs)
                
            else:
                # Tabular Model Loading
                if base_name == "random_forest":
                    from src.models.baselines import RandomForestBaseline
                    model = RandomForestBaseline(config)
                elif base_name == "logistic_regression":
                    from src.models.baselines import LogisticBaseline
                    model = LogisticBaseline(config)
                elif base_name == "xgboost":
                    from src.models.xgb import XGBoostModel
                    model = XGBoostModel(config)
                    
                model.load(model_dir / "model.joblib")
                val_preds_dict[base_name] = model.predict_proba(val_ds_tab.ctx_matrix)

        # Train Meta-Learner on the loaded Validation Set probabilities
        from src.models.stacking import StackingMetaLearner
        
        stacker_artifact_dir = base_artifact_dir / "stacking"
        stacker_artifact_dir.mkdir(parents=True, exist_ok=True)
        
        stacker = StackingMetaLearner(config, base_artifact_dir)
        stacker.fit(val_preds_dict, val_ds_tab.y_vector)
        stacker.save()
        
        print(f"\n>>> Stacking Engine execution complete. Meta-learner saved to {stacker_artifact_dir}")

    # ---------------------------------------------------------
    # Phase 5: TEST SET EVALUATION
    # ---------------------------------------------------------
    if config.pipeline.run_evaluation:
        print("\n>>> Phase 5: Evaluation & Artifact Logging")
        from src.evaluation import Evaluator
        
        test_preds_dict = {} 
        y_true_universal = test_ds_tab.y_vector # Ground truth is identical across all models
        
        # 5.1 Evaluate Base Models (if any were trained in THIS run)
        for model_name, model in trained_models.items():
            print(f"\n--- Evaluating {model_name} on Test Set ---")
            
            if model_name == "lstm":
                model.eval()
                test_probs = []
                device = next(model.parameters()).device
                loader = torch.utils.data.DataLoader(test_ds_lstm, batch_size=256, shuffle=False)
                
                with torch.no_grad():
                    for seq_a, seq_b, ctx, _ in loader:
                        seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                        probs = torch.sigmoid(model(seq_a, seq_b, ctx)).cpu().numpy().flatten()
                        test_probs.extend(probs)
                        
                test_preds_dict[model_name] = np.array(test_probs)
                
                lstm_seq_len = config.models.lstm.architecture.seq_len
                indices = np.random.choice(len(train_ds_lstm), min(50, len(train_ds_lstm)), replace=False)
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

                evaluator = Evaluator(base_artifact_dir / model_name)
                evaluator.generate_report(y_true_universal, test_preds_dict[model_name], model=model, X_train=X_train_shap, model_name="lstm", feature_names=all_names)

            else:
                test_preds_dict[model_name] = model.predict_proba(test_ds_tab.ctx_matrix)
                
                ctx_feature_names = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]
                indices = np.random.choice(len(train_ds_tab), min(500, len(train_ds_tab)), replace=False)
                X_train_df = pd.DataFrame(train_ds_tab.ctx_matrix[indices], columns=ctx_feature_names)

                evaluator = Evaluator(base_artifact_dir / model_name)
                evaluator.generate_report(y_true_universal, test_preds_dict[model_name], model=model, X_train=X_train_df, model_name=model_name, feature_names=ctx_feature_names)

        # 5.2 Evaluate Stacking Model (Decoupled Execution)
        if "stacking" in config.pipeline.models_to_train:
            print(f"\n--- Evaluating stacking on Test Set ---")
            from src.models.stacking import StackingMetaLearner
            
            stacker_artifact_dir = base_artifact_dir / "stacking"
            stacker = StackingMetaLearner(config, base_artifact_dir).load()
            source_dir = Path(config.pipeline.stacking_base_artifact_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Extract Test probabilities for all required base models
            for base_name in stacker.model_names:
                if base_name not in test_preds_dict:
                    print(f"    > Generating Test Set predictions for {base_name}...")
                    model_dir = source_dir / base_name
                    
                    if base_name == "lstm":
                        from src.models.nn import SiameseLSTM
                        with open(model_dir / "hyperparameters.json", "r") as f:
                            seq_len = json.load(f)["architecture"]["seq_len"]
                        
                        model = SiameseLSTM(config, test_ds_lstm.seq_matrix.shape[1], test_ds_lstm.ctx_matrix.shape[1]).to(device)
                        model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
                        model.eval()
                        
                        test_probs = []
                        loader = torch.utils.data.DataLoader(test_ds_lstm, batch_size=256, shuffle=False)
                        with torch.no_grad():
                            for seq_a, seq_b, ctx, _ in loader:
                                seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                                test_probs.extend(torch.sigmoid(model(seq_a, seq_b, ctx)).cpu().numpy().flatten())
                        test_preds_dict[base_name] = np.array(test_probs)
                        
                    else:
                        if base_name == "random_forest":
                            from src.models.baselines import RandomForestBaseline
                            model = RandomForestBaseline(config)
                        elif base_name == "logistic_regression":
                            from src.models.baselines import LogisticBaseline
                            model = LogisticBaseline(config)
                        elif base_name == "xgboost":
                            from src.models.xgb import XGBoostModel
                            model = XGBoostModel(config)
                            
                        model.load(model_dir / "model.joblib")
                        test_preds_dict[base_name] = model.predict_proba(test_ds_tab.ctx_matrix)
                        
            # Execute Stacked Predictions
            stacked_preds = stacker.predict_proba(test_preds_dict)
            
            # Meta-Learner Evaluation
            evaluator = Evaluator(stacker_artifact_dir)
            
            # Provide a mocked background dataset for SHAP to explain the meta-learner
            X_train_meta = pd.DataFrame(
                np.column_stack([test_preds_dict[name] for name in stacker.model_names])[:100], 
                columns=stacker.model_names
            )
            
            evaluator.generate_report(
                y_true_universal, 
                stacked_preds, 
                model=stacker.meta_model, 
                X_train=X_train_meta, 
                model_name="stacking", 
                feature_names=stacker.model_names
            )

    print(f"\n>>> Global Execution Complete. All artifacts saved to: {base_artifact_dir}")

if __name__ == "__main__":
    main()