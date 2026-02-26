import os
import redis
import logging
import pandas as pd
from pathlib import Path
from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStoreManager:
    def __init__(self, host=None, port=6379):
        # Dynamically route to the Redis container on the custom network
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.r = redis.Redis(host=self.host, port=port, decode_responses=True)
        self.engineer = FeatureEngineer(rolling_window=10)

    def seed_database(self, raw_data_dir: Path):
        logger.info(f"Connecting to Redis at {self.host}...")
        if not self.r.ping():
            raise ConnectionError(f"Redis instance at {self.host} unreachable.")

        logger.info("Loading historical ledger...")
        files = sorted(list(raw_data_dir.glob("atp_matches_*.csv")))[-2:]
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        
        logger.info("Executing stateful feature engineering...")
        engineered_df = self.engineer.generate_features(df)
        
        logger.info("Extracting latest player states...")
        latest_states = engineered_df.sort_values('tourney_date').groupby('player').tail(1)
        
        logger.info("Pushing to Redis Feature Store...")
        pipeline = self.r.pipeline()
        for _, row in latest_states.iterrows():
            player_name = row['player']
            # DEFENSIVE OVERRIDE: Pandas native JSON conversion handles Timestamps
            state_json = row.dropna().to_json(date_format='iso')
            redis_key = f"player:{player_name}:state"
            pipeline.set(redis_key, state_json)
            
        pipeline.execute()
        logger.info(f"Successfully seeded {len(latest_states)} player profiles into Redis.")

if __name__ == "__main__":
    manager = FeatureStoreManager()
    manager.seed_database(Path("/app/data/raw"))