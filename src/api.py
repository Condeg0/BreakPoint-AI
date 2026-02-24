import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ValidationError

# Assuming internal modules exist based on your system context.
# Adjust import paths strictly to your package structure.
from src.features import FeatureEngineer
from src.inference import MetaLearnerPipeline

# Configure system-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Tennis Forecast Inference API",
    description="RESTful interface for Siamese LSTM and Stacking Meta-Learner predictions.",
    version="1.0.0"
)

# Load stateful objects globally to persist across requests.
# A failure here intentionally crashes the container on startup.
try:
    feature_engineer = FeatureEngineer.load_state()
    model_pipeline = MetaLearnerPipeline.load_frozen_model()
    logger.info("Pipeline and feature engineer loaded successfully.")
except Exception as e:
    logger.critical(f"System initialization failure: {e}")
    raise RuntimeError("Failed to load ML artifacts.") from e

# ---------------------------------------------------------
# Payload Schemas (Strict Typing)
# ---------------------------------------------------------

class PlayerPayload(BaseModel):
    player_id: int = Field(..., description="Unique ATP player identifier")
    rank: int = Field(..., gt=0, description="Current ATP rank")
    recent_form_index: float = Field(..., ge=0.0, le=1.0, description="Normalized recent win rate or ELO form")

class MatchPredictionRequest(BaseModel):
    surface_id: int = Field(..., description="Encoded surface type (e.g., 0=Hard, 1=Clay, 2=Grass)")
    tournament_level: int = Field(..., description="Encoded tournament tier")
    player_1: PlayerPayload
    player_2: PlayerPayload

class MatchPredictionResponse(BaseModel):
    player_1_win_probability: float
    player_2_win_probability: float
    confidence_score: float

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.post(
    "/predict", 
    response_model=MatchPredictionResponse, 
    status_code=status.HTTP_200_OK
)
async def predict_match(payload: MatchPredictionRequest) -> Dict[str, float]:
    """
    Routes an upcoming match payload through the feature engineering pipeline 
    and the frozen stacking meta-learner.
    """
    try:
        # 1. Extract strict dictionary from validated Pydantic model
        raw_data = payload.model_dump()

        # 2. Execute stateful feature engineering (temporal/spatial transformations)
        engineered_features = feature_engineer.transform(raw_data)

        # 3. Execute inference via the Stacking Meta-Learner
        probabilities = model_pipeline.predict_proba(engineered_features)

        return {
            "player_1_win_probability": round(probabilities["p1"], 4),
            "player_2_win_probability": round(probabilities["p2"], 4),
            "confidence_score": round(abs(probabilities["p1"] - probabilities["p2"]), 4)
        }

    except ValueError as ve:
        # Catches feature engineering shape mismatches or logic errors
        logger.error(f"Data transformation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Malformed feature data: {str(ve)}"
        )
    except Exception as e:
        # Catches inference engine failures, memory issues, or unhandled exceptions
        logger.error(f"Inference failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal system error during model inference."
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness probe for container orchestration."""
    return {"status": "healthy", "artifacts_loaded": True}