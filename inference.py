import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.config import ProjectConfig
from src.features import FeatureEngineer
from src.data import Preprocessor, TennisDataset
from src.models.nn import SiameseLSTM
from src.models.baselines import RandomForestBaseline, LogisticBaseline

def load_recent_history(raw_dir: Path, years_back=2) -> pd.DataFrame:
    """Loads only the most recent historical data required for rolling windows."""
    files = sorted(list(raw_dir.glob("atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw data found in {raw_dir}")
    
    recent_files = files[-years_back:]
    print(f"Loading historical context from: {[f.name for f in recent_files]}")
    
    dfs = [pd.read_csv(f) for f in recent_files]
    df = pd.concat(dfs, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d", errors='coerce')
    df = df.dropna(subset=['tourney_date']).sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="BreakPoint AI: Inference Engine")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="lstm", help="Specific model to use for inference")
    args = parser.parse_args()

    config = ProjectConfig.load(args.config)

    # 1. Validation
    if not config.pipeline.inference_artifact_dir or not config.pipeline.inference_input_file:
        raise ValueError("Inference artifact directory and input file must be specified in config.yaml.")

    artifact_dir = Path(config.pipeline.inference_artifact_dir)
    input_file = Path(config.pipeline.inference_input_file)
    output_file = Path(config.pipeline.inference_output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Inference input file not found: {input_file}")

    # 2. The "Stitching" Mechanism
    print("\n>>> Phase 1: Context Stitching")
    history_df = load_recent_history(Path(config.data.paths.raw_dir))
    
    # Load mathces to predict and enforce date format
    inference_df = pd.read_csv(input_file)
    inference_df['tourney_date'] = pd.to_datetime(inference_df['tourney_date'], format="%Y%m%d", errors='coerce')
    
    # Tag inference rows so we can extract them later
    inference_df['is_inference'] = True
    history_df['is_inference'] = False

    # Combined data: (historic + to be predicted)
    combined_df = pd.concat([history_df, inference_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=['tourney_date', 'match_num', 'winner_name', 'loser_name'], 
        keep='last'
    )
    combined_df = combined_df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    print("\n>>> Phase 2: Feature Engineering (Stateful)")
    engineer = FeatureEngineer(rolling_window=10)
    feat_df = engineer.generate_features(combined_df)

    # Slice out ONLY the inference matches
    target_df = feat_df[feat_df['is_inference'] == True].copy()
    target_df = target_df.drop(columns=['is_inference']).reset_index(drop=True)
    
    if len(target_df) == 0:
        raise ValueError("Feature Engineering stripped the inference rows. Check date ordering or missing critical columns.")

    print(f"Processed {len(target_df)} upcoming matches for inference.")

    # 3. Artifact Loading
    print(f"\n>>> Phase 3: Artifact Loading ({args.model})")
    preprocessor = Preprocessor(config).load(artifact_dir / "global_preprocessor.pkl")
    model_dir = artifact_dir / args.model

    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifacts not found at {model_dir}")

    # 4. Dataset & Model Preparation
    print("\n>>> Phase 4: Prediction Execution")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "lstm":
        with open(model_dir / "hyperparameters.json", "r") as f:
            hyperparams = json.load(f)
        
        seq_len = hyperparams["architecture"]["seq_len"]
        ds = TennisDataset(target_df, preprocessor, mode="lstm", seq_len=seq_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

        model = SiameseLSTM(config, ds.seq_matrix.shape[1], ds.ctx_matrix.shape[1]).to(device)
        model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
        model.eval()

        predictions = []
        with torch.no_grad():
            for seq_a, seq_b, ctx, _ in loader:
                seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                logits = model(seq_a, seq_b, ctx)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                predictions.extend(probs)

    elif args.model in ["random_forest", "logistic_regression", "xgboost"]:
        ds = TennisDataset(target_df, preprocessor, mode="tabular")
        
        if args.model == "random_forest":
            model = RandomForestBaseline(config)
        elif args.model == "logistic_regression":
            model = LogisticBaseline(config)
        elif args.model == "xgboost":
            from src.models.xgb import XGBoostModel
            model = XGBoostModel(config)
            
        model.load(model_dir / "model.joblib")
        predictions = model.predict_proba(ds.ctx_matrix)

    elif args.model == "stacking":
        from src.models.stacking import StackingMetaLearner
        
        # Load the Meta-Learner first to know which base models are required
        stacker = StackingMetaLearner(config, artifact_dir).load()
        base_preds = {}
        
        print(f"Executing base models for stacking ensemble: {stacker.model_names}")
        for base_name in stacker.model_names:
            model_dir = artifact_dir / base_name
            
            if base_name == "lstm":
                with open(model_dir / "hyperparameters.json", "r") as f:
                    seq_len = json.load(f)["architecture"]["seq_len"]
                ds = TennisDataset(target_df, preprocessor, mode="lstm", seq_len=seq_len)
                loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
                
                model = SiameseLSTM(config, ds.seq_matrix.shape[1], ds.ctx_matrix.shape[1]).to(device)
                model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
                model.eval()
                
                probs = []
                with torch.no_grad():
                    for seq_a, seq_b, ctx, _ in loader:
                        seq_a, seq_b, ctx = seq_a.to(device), seq_b.to(device), ctx.to(device)
                        probs.extend(torch.sigmoid(model(seq_a, seq_b, ctx)).cpu().numpy().flatten())
                base_preds[base_name] = np.array(probs)
                
            else:
                ds = TennisDataset(target_df, preprocessor, mode="tabular")
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
                base_preds[base_name] = model.predict_proba(ds.ctx_matrix)
                
        # Generate final ensemble prediction
        predictions = stacker.predict_proba(base_preds)

    else:
        raise ValueError(f"Inference not implemented for model: {args.model}")
    


    # 5. Output
    results = target_df[['tourney_date', 'player', 'opponent', 'surface']].copy()
    results[f'prob_player_wins_{args.model}'] = predictions

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    
    print(f"\n>>> Inference Complete. Forecasts saved to: {output_file}")
    print(results.head())

if __name__ == "__main__":
    main()