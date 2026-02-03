import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    raw_path: Path
    train_cutoff: str = "2022-12-31"
    test_start: str = "2024-01-01"
    target_col: str = "label"

    # 1. BASELINE / CONTEXT FEATURES (The "Static" Input)
    # Used by Random Forest, LogReg, and as the "Current Context" for LSTM
    features: list[str] = Field(default_factory=lambda: [
        "rank_diff", "rank", "opponent_rank",
        "ace_roll_diff", "df_roll_diff", "win_pct_roll_diff",
        "h2h_win_rate", "days_since"
    ])

    # 2. SEQUENCE FEATURES (The "Dynamic" History)
    # Used ONLY by LSTM to build the history tensor.
    # These are raw stats (e.g., "ace", "df") from PAST matches.
    sequence_features: list[str] = Field(default_factory=lambda: [
        "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
        "rank", "winner_rank", "loser_rank" # raw rank at that time
    ])

    drop_cols: list[str] = Field(default_factory=lambda: [
        "match_id", "match_uid", "player", "opponent",
        "match_num", "tourney_date", "label", "winner_name", "loser_name", "id"
    ])

    cat_cols: list[str] = Field(default_factory=lambda: ["surface", "tourney_level", "round"])

class ModelConfig(BaseModel):
    name: str
    # Hyperparams
    seq_len: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.4
    n_estimators: int = 200
    max_depth: int = 8

class TrainConfig(BaseModel):
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 0.001
    seed: int = 42
    # New Flag for Optional Tuning
    tuning: bool = False

class ProjectConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def load(cls, path: Path | str) -> "ProjectConfig":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)
