import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Any, Set, Union, Optional, Self

from src.logger import get_logger
from src.config import ProjectConfig

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, config: ProjectConfig) -> None:
        self.config: ProjectConfig = config
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.feat_map: Dict[str, int] = {}
        self.ctx_indices: List[int] = []
        self.seq_indices: List[int] = []

    def fit(self, df: pd.DataFrame) -> Self:
        ctx_whitelist: Set[str] = set(self.config.data.features.context)
        available: Set[str] = set(df.columns)
        valid_ctx: List[str] = list(ctx_whitelist.intersection(available))

        seq_whitelist: Set[str] = set(self.config.data.features.sequence)
        valid_seq: List[str] = list(seq_whitelist.intersection(available))

        all_numeric: List[str] = sorted(list(set(valid_ctx + valid_seq)))
        valid_cat: List[str] = [c for c in self.config.data.features.categorical if c in available]

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.pipeline = ColumnTransformer([
            ('num', num_pipe, all_numeric),
            ('cat', cat_pipe, valid_cat)
        ])
        self.pipeline.fit(df)

        self.feature_names = [x.split("__")[-1] for x in self.pipeline.get_feature_names_out()]
        self.feat_map = {name: i for i, name in enumerate(self.feature_names)}

        self.ctx_indices = []
        for f in valid_ctx:
            if f in self.feat_map: self.ctx_indices.append(self.feat_map[f])
        
        for cat in valid_cat:
            for name in self.feature_names:
                if name.startswith(f"{cat}_"):
                    self.ctx_indices.append(self.feat_map[name])

        self.seq_indices = []
        for f in valid_seq:
            if f in self.feat_map: self.seq_indices.append(self.feat_map[f])

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet!")
        return self.pipeline.transform(df)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    def load(self, path: Path) -> Preprocessor:
        loaded: Preprocessor = joblib.load(path)
        self.pipeline = loaded.pipeline
        self.feature_names = loaded.feature_names
        self.feat_map = loaded.feat_map
        self.ctx_indices = loaded.ctx_indices
        self.seq_indices = loaded.seq_indices
        self.config = loaded.config
        return self

class TennisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: Preprocessor, mode: str = "tabular", seq_len: int = 10) -> None:
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.preprocessor: Preprocessor = preprocessor
        self.mode: str = mode
        self.seq_len: int = seq_len

        self.full_matrix: np.ndarray = self.preprocessor.transform(self.df).astype(np.float32)
        self.ctx_matrix: np.ndarray = self.full_matrix[:, self.preprocessor.ctx_indices]
        self.seq_matrix: np.ndarray = self.full_matrix[:, self.preprocessor.seq_indices]

        self.dates: np.ndarray = self.df['tourney_date'].values.astype('datetime64[D]').astype(np.int64)

        target_col: str = self.preprocessor.config.data.features.target
        if target_col in self.df.columns:
            self.y_vector: np.ndarray = self.df[target_col].values.astype(np.float32)
        else:
            self.y_vector = np.zeros(len(self.df), dtype=np.float32)

        if self.mode == "lstm":
            self.player_history: Dict[str, np.ndarray] = self._build_history_index()

    def _build_history_index(self) -> Dict[str, np.ndarray]:
        history: Dict[str, np.ndarray] = {}
        groups = self.df.groupby('player')
        for player, group in groups:
            history[player] = group.index.to_numpy()
        return history

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, np.float32], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.mode == "tabular":
            return self.ctx_matrix[idx], self.y_vector[idx]

        row: pd.Series = self.df.iloc[idx]
        current_date: np.int64 = self.dates[idx]

        hist_A: np.ndarray = self._get_sequence(row['player'], current_date)
        hist_B: np.ndarray = self._get_sequence(row['opponent'], current_date)
        current_ctx: np.ndarray = self.ctx_matrix[idx]

        return (torch.tensor(hist_A),
                torch.tensor(hist_B),
                torch.tensor(current_ctx),
                torch.tensor(self.y_vector[idx]))

    def _get_sequence(self, player: str, current_date: np.int64) -> np.ndarray:
        if player not in self.player_history:
            return np.zeros((self.seq_len, self.seq_matrix.shape[1]), dtype=np.float32)

        all_indices: np.ndarray = self.player_history[player]
        candidate_dates: np.ndarray = self.dates[all_indices]
        mask: np.ndarray = candidate_dates < current_date
        past_indices: np.ndarray = all_indices[mask]

        if len(past_indices) == 0:
            return np.zeros((self.seq_len, self.seq_matrix.shape[1]), dtype=np.float32)

        selected_indices: np.ndarray = past_indices[-self.seq_len:]
        seq_data: np.ndarray = self.seq_matrix[selected_indices]

        if len(seq_data) < self.seq_len:
            pad_len: int = self.seq_len - len(seq_data)
            padding: np.ndarray = np.zeros((pad_len, self.seq_matrix.shape[1]), dtype=np.float32)
            seq_data = np.vstack([padding, seq_data])

        return seq_data

def load_raw_merged(data_dir: Path) -> pd.DataFrame:
    files: List[Path] = sorted(list(data_dir.glob("atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No 'atp_matches_*.csv' files found in {data_dir}")

    logger.info(f"Merging {len(files)} raw CSV files...")
    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df: pd.DataFrame = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    full_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    return full_df

def load_and_split(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path: Path = Path(config.data.paths.raw_dir)
    df: pd.DataFrame
    if raw_path.is_dir():
        df = load_raw_merged(raw_path)
    else:
        df = pd.read_csv(raw_path)

    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d", errors='coerce')
    df = df.dropna(subset=['tourney_date']).sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    if 'tourney_name' in df.columns:
        mask: pd.Series = ~df['tourney_name'].str.contains("Davis Cup|Laver Cup", case=False, na=False)
        df = df[mask]

    train_cutoff: str = config.data.temporal_splits.train_cutoff
    test_start: str = config.data.temporal_splits.test_start

    train: pd.DataFrame = df[df['tourney_date'] <= train_cutoff].copy()
    mask_val: pd.Series = (df['tourney_date'] > train_cutoff) & (df['tourney_date'] < test_start)
    val: pd.DataFrame = df[mask_val].copy()
    test: pd.DataFrame = df[df['tourney_date'] >= test_start].copy()

    logger.info(f"Data Splitting Complete: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test