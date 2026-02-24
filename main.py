import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Tuple

from src.config import ProjectConfig
from src.features import FeatureEngineer
from src.data import load_and_split, TennisDataset, Preprocessor
from src.training import Trainer
from src.logger import get_logger

def main() -> None:
    # Initialize with the specific run's artifact directory
    logger: logging.Logger = get_logger(__name__, artifact_dir="outputs/run_2026_02_20")

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="BreakPoint AI: Orchestrator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args: argparse.Namespace = parser.parse_args()

    config_path: Path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from {config_path}...")
    config: ProjectConfig = ProjectConfig.load(config_path)

    logger.info("\n>>> Phase 1: Data Ingestion")
    train_raw: pd.DataFrame
    val_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_raw, val_raw, test_raw = load_and_split(config)

    logger.info("\n>>> Phase 2: Feature Engineering")
    full_df: pd.DataFrame = pd.concat([train_raw, val_raw, test_raw], axis=0).sort_values("tourney_date")
    engineer: FeatureEngineer = FeatureEngineer(rolling_window=10)
    full_feat_df: pd.DataFrame = engineer.generate_features(full_df)

    train_cutoff: pd.Timestamp = pd.to_datetime(config.data.temporal_splits.train_cutoff)
    test_start: pd.Timestamp = pd.to_datetime(config.data.temporal_splits.test_start)

    train_df: pd.DataFrame = full_feat_df[full_feat_df['tourney_date'] <= train_cutoff]
    val_df: pd.DataFrame = full_feat_df[(full_feat_df['tourney_date'] > train_cutoff) & (full_feat_df['tourney_date'] < test_start)]
    test_df: pd.DataFrame = full_feat_df[full_feat_df['tourney_date'] >= test_start]

    logger.info(f"Featured Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    logger.info("\n>>> Phase 3: Preprocessing Setup")
    run_id: str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_artifact_dir: Path = Path(config.data.paths.artifact_dir) / run_id
    base_artifact_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor: Preprocessor = Preprocessor(config)
    
    if config.pipeline.models_to_train == ["stacking"]:
        logger.info("    > Stacking Mode: Loading frozen preprocessor to guarantee feature parity.")
        source_dir: Path = Path(config.pipeline.stacking_base_artifact_dir)
        preprocessor.load(source_dir / "global_preprocessor.pkl")
        preprocessor.save(base_artifact_dir / "global_preprocessor.pkl")
    else:
        preprocessor.fit(train_df)
        preprocessor.save(base_artifact_dir / "global_preprocessor.pkl")

    train_ds_tab: TennisDataset = TennisDataset(train_df, preprocessor, mode="tabular")
    val_ds_tab: TennisDataset   = TennisDataset(val_df, preprocessor, mode="tabular")
    test_ds_tab: TennisDataset  = TennisDataset(test_df, preprocessor, mode="tabular")

    lstm_seq_len: int = config.models.lstm.architecture.seq_len
    train_ds_lstm: TennisDataset = TennisDataset(train_df, preprocessor, mode="lstm", seq_len=lstm_seq_len)
    val_ds_lstm: TennisDataset   = TennisDataset(val_df, preprocessor, mode="lstm", seq_len=lstm_seq_len)
    test_ds_lstm: TennisDataset  = TennisDataset(test_df, preprocessor, mode="lstm", seq_len=lstm_seq_len)

    logger.info("\n>>> Phase 4: Model Execution")
    trained_models: Dict[str, Any] = {}

    base_models: List[str] = [m for m in config.pipeline.models_to_train if m != "stacking"]
    
    for model_name in base_models:
        logger.info(f"\n--- Initiating Base Model Run: {model_name} ---")
        
        model_artifact_dir: Path = base_artifact_dir / model_name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)
        trainer: Trainer = Trainer(config, model_artifact_dir)

        model: Any
        if model_name == "lstm":
            model = trainer.train(train_ds_lstm, val_ds_lstm, model_name=model_name)
        else:
            model = trainer.train(train_ds_tab, val_ds_tab, model_name=model_name)
            
        trained_models[model_name] = model

        params_to_save: Dict[str, Any] = {}
        if model_name == "lstm":
            params_to_save["architecture"] = config.models.lstm.architecture.model_dump()
            params_to_save["training"] = config.models.lstm.training.model_dump()
        else:
            model_cfg: Any = getattr(config.models, model_name)
            params_to_save = model_cfg.hyperparameters

        with open(model_artifact_dir / "hyperparameters.json", "w") as f:
            json.dump(params_to_save, f, indent=4)

    if "stacking" in config.pipeline.models_to_train:
        logger.info("\n" + "="*50)
        logger.info(">>>  INITIATING STACKING META-LEARNER (FROM ARTIFACTS)")
        logger.info("="*50)
        
        if not config.pipeline.stacking_base_artifact_dir:
            raise ValueError("stacking_base_artifact_dir must be defined in config.yaml to train the Stacker.")
            
        source_dir = Path(config.pipeline.stacking_base_artifact_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Base artifact directory not found: {source_dir}")

        val_preds_dict: Dict[str, np.ndarray] = {}
        expected_bases: List[str] = ["lstm", "xgboost", "random_forest", "logistic_regression"]
        found_bases: List[str] = [m for m in expected_bases if (source_dir / m).exists()]
        
        if not found_bases:
            raise ValueError(f"No pre-trained base models found in {source_dir}")
            
        logger.info(f"Loading pre-trained base models for Meta-Feature generation: {found_bases}")
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for base_name in found_bases:
            model_dir: Path = source_dir / base_name
            logger.info(f"    > Generating Validation OOF predictions for {base_name}...")
            
            if base_name == "lstm":
                from src.models.nn import SiameseLSTM
                
                with open(model_dir / "hyperparameters.json", "r") as f:
                    seq_len: int = json.load(f)["architecture"]["seq_len"]
                
                model = SiameseLSTM(config, train_ds_lstm.seq_matrix.shape[1], train_ds_lstm.ctx_matrix.shape[1]).to(device)
                model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
                model.eval()
                
                val_probs: List[float] = []
                loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(val_ds_lstm, batch_size=256, shuffle=False)
                with torch.no_grad():
                    for seq_a, seq_b, ctx, _ in loader:
                        seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                        probs: np.ndarray = torch.sigmoid(model(seq_a, seq_b, ctx)).cpu().numpy().flatten()
                        val_probs.extend(probs)
                val_preds_dict[base_name] = np.array(val_probs)
                
            else:
                model_instance: Any
                if base_name == "random_forest":
                    from src.models.baselines import RandomForestBaseline
                    model_instance = RandomForestBaseline(config)
                elif base_name == "logistic_regression":
                    from src.models.baselines import LogisticBaseline
                    model_instance = LogisticBaseline(config)
                elif base_name == "xgboost":
                    from src.models.xgb import XGBoostModel
                    model_instance = XGBoostModel(config)
                    
                model_instance.load(model_dir / "model.joblib")
                val_preds_dict[base_name] = model_instance.predict_proba(val_ds_tab.ctx_matrix)

        from src.models.stacking import StackingMetaLearner
        
        stacker_artifact_dir: Path = base_artifact_dir / "stacking"
        stacker_artifact_dir.mkdir(parents=True, exist_ok=True)
        
        stacker: StackingMetaLearner = StackingMetaLearner(config, base_artifact_dir)
        stacker.fit(val_preds_dict, val_ds_tab.y_vector)
        stacker.save()
        
        logger.info(f"\n>>> Stacking Engine execution complete. Meta-learner saved to {stacker_artifact_dir}")

    if config.pipeline.run_evaluation:
        logger.info("\n>>> Phase 5: Evaluation & Artifact Logging")
        from src.evaluation import Evaluator
        
        test_preds_dict: Dict[str, np.ndarray] = {} 
        y_true_universal: np.ndarray = test_ds_tab.y_vector
        
        for model_name, model_eval in trained_models.items():
            logger.info(f"\n--- Evaluating {model_name} on Test Set ---")
            
            if model_name == "lstm":
                model_eval.eval()
                test_probs: List[float] = []
                device_eval: torch.device = next(model_eval.parameters()).device
                loader_eval: torch.utils.data.DataLoader = torch.utils.data.DataLoader(test_ds_lstm, batch_size=256, shuffle=False)
                
                with torch.no_grad():
                    for seq_a, seq_b, ctx, _ in loader_eval:
                        seq_a, seq_b, ctx = seq_a.to(device_eval), seq_b.to(device_eval), ctx.to(device_eval)
                        probs_eval: np.ndarray = torch.sigmoid(model_eval(seq_a, seq_b, ctx)).cpu().numpy().flatten()
                        test_probs.extend(probs_eval)
                        
                test_preds_dict[model_name] = np.array(test_probs)
                
                lstm_seq_len_eval: int = config.models.lstm.architecture.seq_len
                indices: np.ndarray = np.random.choice(len(train_ds_lstm), min(50, len(train_ds_lstm)), replace=False)
                X_train_shap: np.ndarray = np.array([
                    np.concatenate([train_ds_lstm[i][0].numpy().flatten(), 
                                    train_ds_lstm[i][1].numpy().flatten(), 
                                    train_ds_lstm[i][2].numpy().flatten()])
                    for i in indices
                ])
                
                seq_feats: List[str] = config.data.features.sequence
                ctx_feats: List[str] = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]
                all_names: List[str] = [f"P1_{f}_t-{lstm_seq_len_eval - i}" for i in range(lstm_seq_len_eval) for f in seq_feats] + \
                                       [f"P2_{f}_t-{lstm_seq_len_eval - i}" for i in range(lstm_seq_len_eval) for f in seq_feats] + \
                                       ctx_feats

                evaluator: Evaluator = Evaluator(base_artifact_dir / model_name)
                evaluator.generate_report(y_true_universal, test_preds_dict[model_name], model=model_eval, X_train=X_train_shap, model_name="lstm", feature_names=all_names)

            else:
                test_preds_dict[model_name] = model_eval.predict_proba(test_ds_tab.ctx_matrix)
                
                ctx_feature_names: List[str] = [preprocessor.feature_names[i] for i in preprocessor.ctx_indices]
                indices_eval: np.ndarray = np.random.choice(len(train_ds_tab), min(500, len(train_ds_tab)), replace=False)
                X_train_df: pd.DataFrame = pd.DataFrame(train_ds_tab.ctx_matrix[indices_eval], columns=ctx_feature_names)

                evaluator: Evaluator = Evaluator(base_artifact_dir / model_name)
                evaluator.generate_report(y_true_universal, test_preds_dict[model_name], model=model_eval, X_train=X_train_df, model_name=model_name, feature_names=ctx_feature_names)

        if "stacking" in config.pipeline.models_to_train:
            logger.info(f"\n--- Evaluating stacking on Test Set ---")
            from src.models.stacking import StackingMetaLearner
            
            stacker_artifact_dir_eval: Path = base_artifact_dir / "stacking"
            stacker_eval: StackingMetaLearner = StackingMetaLearner(config, base_artifact_dir).load()
            source_dir_eval: Path = Path(config.pipeline.stacking_base_artifact_dir)
            device_eval_stack: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for base_name_eval in stacker_eval.model_names:
                if base_name_eval not in test_preds_dict:
                    logger.info(f"    > Generating Test Set predictions for {base_name_eval}...")
                    model_dir_eval: Path = source_dir_eval / base_name_eval
                    
                    if base_name_eval == "lstm":
                        from src.models.nn import SiameseLSTM
                        with open(model_dir_eval / "hyperparameters.json", "r") as f:
                            seq_len_eval: int = json.load(f)["architecture"]["seq_len"]
                        
                        model_lstm_eval: SiameseLSTM = SiameseLSTM(config, test_ds_lstm.seq_matrix.shape[1], test_ds_lstm.ctx_matrix.shape[1]).to(device_eval_stack)
                        model_lstm_eval.load_state_dict(torch.load(model_dir_eval / "best_model.pt", map_location=device_eval_stack))
                        model_lstm_eval.eval()
                        
                        test_probs_eval: List[float] = []
                        loader_lstm_eval: torch.utils.data.DataLoader = torch.utils.data.DataLoader(test_ds_lstm, batch_size=256, shuffle=False)
                        with torch.no_grad():
                            for seq_a, seq_b, ctx, _ in loader_lstm_eval:
                                seq_a, seq_b, ctx = seq_a.to(device_eval_stack), seq_b.to(device_eval_stack), ctx.to(device_eval_stack)
                                test_probs_eval.extend(torch.sigmoid(model_lstm_eval(seq_a, seq_b, ctx)).cpu().numpy().flatten())
                        test_preds_dict[base_name_eval] = np.array(test_probs_eval)
                        
                    else:
                        model_instance_eval: Any
                        if base_name_eval == "random_forest":
                            from src.models.baselines import RandomForestBaseline
                            model_instance_eval = RandomForestBaseline(config)
                        elif base_name_eval == "logistic_regression":
                            from src.models.baselines import LogisticBaseline
                            model_instance_eval = LogisticBaseline(config)
                        elif base_name_eval == "xgboost":
                            from src.models.xgb import XGBoostModel
                            model_instance_eval = XGBoostModel(config)
                            
                        model_instance_eval.load(model_dir_eval / "model.joblib")
                        test_preds_dict[base_name_eval] = model_instance_eval.predict_proba(test_ds_tab.ctx_matrix)
                        
            stacked_preds: np.ndarray = stacker_eval.predict_proba(test_preds_dict)
            
            evaluator_stack: Evaluator = Evaluator(stacker_artifact_dir_eval)
            
            X_train_meta: pd.DataFrame = pd.DataFrame(
                np.column_stack([test_preds_dict[name] for name in stacker_eval.model_names])[:100], 
                columns=stacker_eval.model_names
            )
            
            evaluator_stack.generate_report(
                y_true_universal, 
                stacked_preds, 
                model=stacker_eval.meta_model, 
                X_train=X_train_meta, 
                model_name="stacking", 
                feature_names=stacker_eval.model_names
            )

    logger.info(f"\n>>> Global Execution Complete. All artifacts saved to: {base_artifact_dir}")

if __name__ == "__main__":
    main()