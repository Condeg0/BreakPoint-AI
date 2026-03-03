import os
import io
import logging
import pandas as pd
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import psutil


from src.features import FeatureEngineer
from src.inference import MetaLearnerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="BreakPoint AI: Analytical Batch Processor", version="3.0.0")

MODEL_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", "/app/artifacts/prod"))
RAW_DATA_DIR = Path("/app/data/raw") 

try:
    feature_engineer = FeatureEngineer.load_state(base_path=MODEL_DIR)
    model_pipeline = MetaLearnerPipeline.load_frozen_model(base_path=MODEL_DIR)
except Exception as e:
    raise RuntimeError(f"Failed to load ML artifacts: {e}")

class BatchPredictionResponseItem(BaseModel):
    player_1: str
    player_2: str
    player_1_win_probability: float
    player_2_win_probability: float
    confidence_spread: float

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionResponseItem]

class HealthResponse(BaseModel):
    status: str
    memory_usage: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint to report service status and memory usage.
    Crucial for monitoring in memory-constrained environments like Render's free tier.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
    return {"status": "ok", "memory_usage": f"{memory_mb:.2f} MB"}


def load_recent_history(raw_dir: Path, years_back: int = 2) -> pd.DataFrame:
    files = sorted(list(raw_dir.glob("atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"Internal history missing from {raw_dir}. Check Docker build context.")
    dfs = [pd.read_csv(f) for f in files[-years_back:]]
    df = pd.concat(dfs, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'].astype(str), format="%Y%m%d", errors='coerce')
    return df.dropna(subset=['tourney_date']).sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        # 1. Parse Uploaded CSV
        contents = await file.read()
        inference_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        inference_df['is_inference'] = True
        
        rename_map = {
            "player": "winner_name", "opponent": "loser_name", 
            "player_id": "winner_id", "opponent_id": "loser_id", 
            "rank": "winner_rank", "opponent_rank": "loser_rank"
        }
        inference_df = inference_df.rename(columns=rename_map)
        
        # Inject dummy match stats for the future matches to satisfy feature engineering
        stats_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_bpSaved', 'w_bpFaced',
                      'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced']
        for col in stats_cols:
            if col not in inference_df.columns:
                inference_df[col] = 0.0

        # 2. Stitch Context with Internal Ledger
        history_df = load_recent_history(RAW_DATA_DIR)
        history_df['is_inference'] = False
        
        combined_df = pd.concat([history_df, inference_df], ignore_index=True)
        combined_df['tourney_date'] = pd.to_datetime(combined_df['tourney_date'].astype(str), format="%Y%m%d", errors='coerce')
        combined_df = combined_df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

        # 3. Calculate True Rolling Features
        engineered_df = feature_engineer.generate_features(combined_df)
        
        # 4. Execute Batch Inference
        predictions = model_pipeline.predict_batch(engineered_df)
            
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))