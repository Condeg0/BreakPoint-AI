import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure internal modules use absolute imports relative to src
from src.features import FeatureEngineer
from src.inference import MetaLearnerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tennis Forecast Inference API",
    version="1.1.0"
)

# RECTIFICATION: Default must match the Docker COPY destination: /app/artifacts/prod
# The Path cast is mandatory to enable the / operator in downstream modules.
MODEL_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", "/app/artifacts/prod"))

logger.info(f"System Check: Looking for artifacts in {MODEL_DIR.absolute()}")

if not MODEL_DIR.exists():
    logger.critical(f"Directory not found: {MODEL_DIR.absolute()}")
    raise RuntimeError(f"Artifact directory {MODEL_DIR} does not exist inside the container.")

try:
    # Explicitly passing the Path object
    feature_engineer = FeatureEngineer.load_state(base_path=MODEL_DIR)
    model_pipeline = MetaLearnerPipeline.load_frozen_model(base_path=MODEL_DIR)
    logger.info("ML Components initialized successfully.")
except Exception as e:
    logger.critical(f"Initialization failure at {MODEL_DIR}: {e}")
    raise RuntimeError(f"Failed to load ML artifacts: {e}") from e

# --- Schemas ---
class PlayerState(BaseModel):
    player_id: int
    name: str
    rank: int

class MatchPredictionRequest(BaseModel):
    tourney_date: str
    match_num: int
    surface: str
    tourney_level: str
    round: str
    player_1: PlayerState
    player_2: PlayerState

class MatchPredictionResponse(BaseModel):
    player_1_win_probability: float
    player_2_win_probability: float
    confidence_spread: float

# --- Endpoints ---
@app.post("/predict", response_model=MatchPredictionResponse)
async def predict_match(payload: MatchPredictionRequest):
    try:
        raw_df = pd.DataFrame([{
            "tourney_date": payload.tourney_date,
            "match_num": payload.match_num,
            "surface": payload.surface,
            "tourney_level": payload.tourney_level,
            "round": payload.round,
            "player_id": payload.player_1.player_id,
            "player": payload.player_1.name,
            "rank": payload.player_1.rank,
            "opponent_id": payload.player_2.player_id,
            "opponent": payload.player_2.name,
            "opponent_rank": payload.player_2.rank,
            "is_inference": True
        }])

        engineered_features = feature_engineer.generate_features(raw_df)
        
        # NOTE: This will currently return dummy 0.5/0.5 based on your inference.py placeholder.
        probabilities = model_pipeline.predict_proba(engineered_features.to_dict(orient="records")[0])

        return {
            "player_1_win_probability": probabilities.get("p1", 0.5),
            "player_2_win_probability": probabilities.get("p2", 0.5),
            "confidence_spread": abs(probabilities.get("p1", 0.5) - probabilities.get("p2", 0.5))
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}