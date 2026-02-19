import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

class DataPaths(BaseModel):
    raw_dir: str
    processed_dir: str
    artifact_dir: str

class TemporalSplits(BaseModel):
    train_cutoff: str
    test_start: str

class FeatureSets(BaseModel):
    context: list[str]
    sequence: list[str]
    categorical: list[str]
    target: str

class DataConfig(BaseModel):
    paths: DataPaths
    temporal_splits: TemporalSplits
    features: FeatureSets

class PipelineConfig(BaseModel):
    models_to_train: list[str]
    use_stacking: bool
    run_evaluation: bool
    inference_artifact_dir: Optional[str] = None
    inference_input_file: Optional[str] = None
    inference_output_file: Optional[str] = None

class LSTMTrain(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    tuning_enabled: bool

class LSTMArch(BaseModel):
    seq_len: int
    hidden_size: int
    num_layers: int
    dropout: float

class LSTMConfig(BaseModel):
    architecture: LSTMArch
    training: LSTMTrain

class XGBoostConfig(BaseModel):
    hyperparameters: dict
    training: dict

class RandomForestConfig(BaseModel):
    hyperparameters: dict
    training: dict

class LogRegConfig(BaseModel):
    hyperparameters: dict
    training: dict

class StackingConfig(BaseModel):
    meta_learner: str
    cv_folds: int

class ModelsConfig(BaseModel):
    lstm: LSTMConfig
    xgboost: XGBoostConfig
    random_forest: RandomForestConfig
    logistic_regression: LogRegConfig
    stacking: StackingConfig

class ProjectConfig(BaseModel):
    project: dict
    data: DataConfig
    pipeline: PipelineConfig
    models: ModelsConfig

    @classmethod
    def load(cls, path: Path | str) -> "ProjectConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
            
        return cls(**cfg_dict)