import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Union

from src.config import ProjectConfig
from src.features import FeatureEngineer
from src.data import Preprocessor, TennisDataset
from src.models.nn import SiameseLSTM
from src.models.baselines import RandomForestBaseline, LogisticBaseline
from src.logger import get_logger

logger: logging.Logger = get_logger(__name__)

def load_recent_history(raw_dir: Path, years_back: int = 2) -> pd.DataFrame:
    """Loads only the most recent historical data required for rolling windows."""
    files: List[Path] = sorted(list(raw_dir.glob("atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw data found in {raw_dir}")
    
    recent_files: List[Path] = files[-years_back:]
    logger.info(f"Loading historical context from: {[f.name for f in recent_files]}")
    
    dfs: List[pd.DataFrame] = [pd.read_csv(f) for f in recent_files]
    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d", errors='coerce')
    df = df.dropna(subset=['tourney_date']).sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
    return df

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="BreakPoint AI: Inference Engine")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="lstm", help="Specific model to use for inference")
    args: argparse.Namespace = parser.parse_args()

    config: ProjectConfig = ProjectConfig.load(Path(args.config))

    if not config.pipeline.inference_artifact_dir or not config.pipeline.inference_input_file:
        logger.error("Inference artifact directory and input file must be specified in config.yaml.")
        sys.exit(1)

    artifact_dir: Path = Path(config.pipeline.inference_artifact_dir)
    global logger
    logger = get_logger(__name__, artifact_dir=artifact_dir)

    input_file: Path = Path(config.pipeline.inference_input_file)
    output_file: Path = Path(config.pipeline.inference_output_file) if config.pipeline.inference_output_file else Path("outputs/inference.csv")

    if not input_file.exists():
        logger.error(f"Inference input file not found: {input_file}")
        sys.exit(1)

    logger.info("\n>>> Phase 1: Context Stitching")
    history_df: pd.DataFrame = load_recent_history(Path(config.data.paths.raw_dir))
    
    inference_df: pd.DataFrame = pd.read_csv(input_file)
    inference_df['tourney_date'] = pd.to_datetime(inference_df['tourney_date'], format="%Y%m%d", errors='coerce')
    
    inference_df['is_inference'] = True
    history_df['is_inference'] = False

    combined_df: pd.DataFrame = pd.concat([history_df, inference_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=['tourney_date', 'match_num', 'winner_name', 'loser_name'], 
        keep='last'
    )
    combined_df = combined_df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    logger.info("\n>>> Phase 2: Feature Engineering (Stateful)")
    engineer: FeatureEngineer = FeatureEngineer(rolling_window=10)
    feat_df: pd.DataFrame = engineer.generate_features(combined_df)

    target_df: pd.DataFrame = feat_df[feat_df['is_inference'] == True].copy()
    target_df = target_df.drop(columns=['is_inference']).reset_index(drop=True)
    
    if len(target_df) == 0:
        logger.warning("Feature Engineering stripped all inference rows. Check date ordering or missing critical columns.")
        sys.exit(0)

    logger.info(f"Processed {len(target_df)} upcoming matches for inference.")

    logger.info(f"\n>>> Phase 3: Artifact Loading ({args.model})")
    preprocessor: Preprocessor = Preprocessor(config).load(artifact_dir / "global_preprocessor.pkl")
    model_dir: Path = artifact_dir / args.model

    if not model_dir.exists():
        logger.error(f"Model artifacts not found at {model_dir}")
        sys.exit(1)

    logger.info("\n>>> Phase 4: Prediction Execution")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: Union[List[float], np.ndarray]

    if args.model == "lstm":
        with open(model_dir / "hyperparameters.json", "r") as f:
            hyperparams: Dict[str, Any] = json.load(f)
        
        seq_len: int = hyperparams["architecture"]["seq_len"]
        ds: TennisDataset = TennisDataset(target_df, preprocessor, mode="lstm", seq_len=seq_len)
        loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

        model_lstm: SiameseLSTM = SiameseLSTM(config, ds.seq_matrix.shape[1], ds.ctx_matrix.shape[1]).to(device)
        model_lstm.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
        model_lstm.eval()

        predictions = []
        with torch.no_grad():
            for seq_a, seq_b, ctx, _ in loader:
                seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                logits: torch.Tensor = model_lstm(seq_a, seq_b, ctx)
                probs: np.ndarray = torch.sigmoid(logits).cpu().numpy().flatten()
                predictions.extend(probs)

    elif args.model in ["random_forest", "logistic_regression", "xgboost"]:
        ds_tabular: TennisDataset = TennisDataset(target_df, preprocessor, mode="tabular")
        model_tabular: Union[RandomForestBaseline, LogisticBaseline, Any]
        
        if args.model == "random_forest":
            model_tabular = RandomForestBaseline(config)
        elif args.model == "logistic_regression":
            model_tabular = LogisticBaseline(config)
        elif args.model == "xgboost":
            from src.models.xgb import XGBoostModel
            model_tabular = XGBoostModel(config)
        else:
             # Should be unreachable due to outer if/else
            logger.error(f"Unknown tabular model: {args.model}")
            sys.exit(1)

        model_tabular.load(model_dir / "model.joblib")
        predictions = model_tabular.predict_proba(ds_tabular.ctx_matrix)

    elif args.model == "stacking":
        from src.models.stacking import StackingMetaLearner
        
        stacker: StackingMetaLearner = StackingMetaLearner(config, artifact_dir).load()
        base_preds: Dict[str, np.ndarray] = {}
        
        logger.info(f"Executing base models for stacking ensemble: {stacker.model_names}")
        for base_name in stacker.model_names:
            base_model_dir: Path = artifact_dir / base_name
            
            if base_name == "lstm":
                with open(base_model_dir / "hyperparameters.json", "r") as f:
                    seq_len_stack: int = json.load(f)["architecture"]["seq_len"]
                ds_lstm_stack: TennisDataset = TennisDataset(target_df, preprocessor, mode="lstm", seq_len=seq_len_stack)
                loader_lstm_stack: torch.utils.data.DataLoader = torch.utils.data.DataLoader(ds_lstm_stack, batch_size=64, shuffle=False)
                
                lstm_model_stack: SiameseLSTM = SiameseLSTM(config, ds_lstm_stack.seq_matrix.shape[1], ds_lstm_stack.ctx_matrix.shape[1]).to(device)
                lstm_model_stack.load_state_dict(torch.load(base_model_dir / "best_model.pt", map_location=device))
                lstm_model_stack.eval()
                
                lstm_probs: List[float] = []
                with torch.no_grad():
                    for seq_a, seq_b, ctx, _ in loader_lstm_stack:
                        seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                        lstm_probs.extend(torch.sigmoid(lstm_model_stack(seq_a, seq_b, ctx)).cpu().numpy().flatten())
                base_preds[base_name] = np.array(lstm_probs)
                
            else:
                ds_tab_stack: TennisDataset = TennisDataset(target_df, preprocessor, mode="tabular")
                tab_model_stack: Union[RandomForestBaseline, LogisticBaseline, Any]
                if base_name == "random_forest":
                    tab_model_stack = RandomForestBaseline(config)
                elif base_name == "logistic_regression":
                    tab_model_stack = LogisticBaseline(config)
                elif base_name == "xgboost":
                    from src.models.xgb import XGBoostModel
                    tab_model_stack = XGBoostModel(config)
                else:
                    logger.error(f"Unknown tabular model for stacking: {base_name}")
                    sys.exit(1)

                tab_model_stack.load(base_model_dir / "model.joblib")
                base_preds[base_name] = tab_model_stack.predict_proba(ds_tab_stack.ctx_matrix)
                
        predictions = stacker.predict_proba(base_preds)

    else:
        logger.error(f"Inference not implemented for model: {args.model}")
        sys.exit(1)

    results: pd.DataFrame = target_df[['tourney_date', 'player', 'opponent', 'surface']].copy()
    results[f'prob_player_wins_{args.model}'] = predictions

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    
    logger.info(f"\n>>> Inference Complete. Forecasts saved to: {output_file}")
    logger.info(f"Inference results head:\n{results.head().to_string()}")

if __name__ == "__main__":
    main()