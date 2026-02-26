import os
import io
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel

# Ensure internal modules use absolute imports relative to src
from src.features import FeatureEngineer
from src.inference import MetaLearnerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tennis Forecast Inference API",
    version="1.2.0"
)

MODEL_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", "/app/artifacts/prod"))

logger.info(f"System Check: Looking for artifacts in {MODEL_DIR.absolute()}")

if not MODEL_DIR.exists():
    logger.critical(f"Directory not found: {MODEL_DIR.absolute()}")
    raise RuntimeError(f"Artifact directory {MODEL_DIR} does not exist inside the container.")

try:
    feature_engineer = FeatureEngineer.load_state(base_path=MODEL_DIR)
    model_pipeline = MetaLearnerPipeline.load_frozen_model(base_path=MODEL_DIR)
    logger.info("ML Components initialized successfully.")
except Exception as e:
    logger.critical(f"Initialization failure at {MODEL_DIR}: {e}")
    raise RuntimeError(f"Failed to load ML artifacts: {e}") from e

class BatchPredictionResponseItem(BaseModel):
    match_id: str
    player_1_win_probability: float
    player_2_win_probability: float
    confidence_spread: float

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionResponseItem]

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for batch prediction.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Mark data for inference handling in feature engineering
        df['is_inference'] = True
        
        # Stateful Feature Generation
        engineered_features = feature_engineer.generate_features(df)
        
        # Extract target predictions
        target_df = engineered_features[engineered_features['is_inference'] == True].copy()
        target_records = target_df.to_dict(orient="records")
        
        predictions = []
        for row in target_records:
            probs = model_pipeline.predict_proba(row)
            
            match_id = row.get("match_uid", "unknown")
            p1 = probs.get("p1", 0.5)
            p2 = probs.get("p2", 0.5)
            
            predictions.append({
                "match_id": str(match_id),
                "player_1_win_probability": p1,
                "player_2_win_probability": p2,
                "confidence_spread": abs(p1 - p2)
            })
            
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}