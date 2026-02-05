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
from typing import Tuple

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.feature_names = []
        # We track indices to split the big matrix later
        self.feat_indices = []
        self.seq_indices = []

    def fit(self, df: pd.DataFrame):

        # 1. Identify "Context" Features (Rolling/Static)
        ctx_whitelist = set(self.config.data.features)
        available = set(df.columns)
        valid_ctx = list(ctx_whitelist.intersection(available))

        # 2. Identify "Sequence" Features (Raw Stats)
        seq_whitelist = set(self.config.data.sequence_features)
        valid_seq = list(seq_whitelist.intersection(available))

        # Combine strictly for fitting the scaler (Union of all needed columns)
        # We sort to ensure deterministic order
        all_numeric = sorted(list(set(valid_ctx + valid_seq)))

        # Categorical features (Static context only usually)
        valid_cat = [c for c in self.config.data.cat_cols if c in available]

        # Define transformations
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

        # Store Feature Names and Indices for lookups later
        self.feature_names = [x.split("__")[-1] for x in self.pipeline.get_feature_names_out()]

        # Map feature names to their index in the transformed matrix
        self.feat_map = {name: i for i, name in enumerate(self.feature_names)}

        # Pre-calculate indices for fast slicing in Dataset
        # We need to know which columns correspond to 'features' vs 'sequence_features'
        # Note: Categorical features are one-hot encoded, so we need to find all their generated columns

        self.ctx_indices = []
        for f in valid_ctx:
             if f in self.feat_map: self.ctx_indices.append(self.feat_map[f])
        # Add encoded categoricals to context
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

    def save(self, path: Path):
        joblib.dump(self, path) # Save whole object to keep indices

    def load(self, path: Path):
        loaded = joblib.load(path)
        self.pipeline = loaded.pipeline
        self.feature_names = loaded.feature_names
        self.feat_map = loaded.feat_map
        self.ctx_indices = loaded.ctx_indices
        self.seq_indices = loaded.seq_indices
        self.config = loaded.config
        return self

class TennisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: Preprocessor, mode: str = "tabular", seq_len: int = 10):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.mode = mode
        self.seq_len = seq_len

        # 1. Transform ALL features into a big matrix
        self.full_matrix = self.preprocessor.transform(self.df).astype(np.float32)

        # 2. Pre-slice for speed
        # Context Matrix: Rows x (Rolling + Static + OneHot)
        self.ctx_matrix = self.full_matrix[:, self.preprocessor.ctx_indices]

        # Sequence Matrix: Rows x (Raw Stats)
        # This contains the raw stats for every match, which we will look up for history
        self.seq_matrix = self.full_matrix[:, self.preprocessor.seq_indices]

        # Store Dates
        self.dates = self.df['tourney_date'].values.astype('datetime64[D]').astype(np.int64)

        # Target
        target_col = self.preprocessor.config.data.target_col
        if target_col in self.df.columns:
            self.y_vector = self.df[target_col].values.astype(np.float32)
        else:
            self.y_vector = np.zeros(len(self.df), dtype=np.float32)

        if self.mode == "lstm":
            self.player_history = self._build_history_index()

    def _build_history_index(self):
        history = {}
        groups = self.df.groupby('player')
        for player, group in groups:
            history[player] = group.index.to_numpy()
        return history

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # MODE 1: TABULAR (Random Forest / LogReg)
        # Returns only the Context Vector (Rolling stats + Metadata)
        if self.mode == "tabular":
            return self.ctx_matrix[idx], self.y_vector[idx]

        # MODE 2: LSTM (Deep Learning)
        # Returns (History_A, History_B, Context, Label)
        row = self.df.iloc[idx]
        current_date = self.dates[idx]

        hist_A = self._get_sequence(row['player'], current_date)
        hist_B = self._get_sequence(row['opponent'], current_date)
        current_ctx = self.ctx_matrix[idx]

        return (torch.tensor(hist_A),
                torch.tensor(hist_B),
                torch.tensor(current_ctx),
                torch.tensor(self.y_vector[idx]))

    def _get_sequence(self, player, current_date):
        if player not in self.player_history:
            return np.zeros((self.seq_len, self.seq_matrix.shape[1]), dtype=np.float32)

        all_indices = self.player_history[player]

        # Vectorized Date Filter (Strict Past)
        candidate_dates = self.dates[all_indices]
        mask = candidate_dates < current_date
        past_indices = all_indices[mask]

        if len(past_indices) == 0:
            return np.zeros((self.seq_len, self.seq_matrix.shape[1]), dtype=np.float32)

        selected_indices = past_indices[-self.seq_len:]

        # LOOKUP in SEQ_MATRIX (Raw Stats)
        seq_data = self.seq_matrix[selected_indices]

        if len(seq_data) < self.seq_len:
            pad_len = self.seq_len - len(seq_data)
            padding = np.zeros((pad_len, self.seq_matrix.shape[1]), dtype=np.float32)
            seq_data = np.vstack([padding, seq_data])

        return seq_data

def load_raw_merged(data_dir: Path) -> pd.DataFrame:
    files = sorted(list(data_dir.glob("atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No 'atp_matches_*.csv' files found in {data_dir}")

    print(f"Merging {len(files)} raw CSV files (excluding base_data.csv)...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def load_and_split(config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path = Path(config.data.raw_path)
    if raw_path.is_dir():
        df = load_raw_merged(raw_path)
    else:
        df = pd.read_csv(raw_path)

    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d", errors='coerce')
    df = df.dropna(subset=['tourney_date']).sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    if 'tourney_name' in df.columns:
        mask = ~df['tourney_name'].str.contains("Davis Cup|Laver Cup", case=False, na=False)
        df = df[mask]

    train = df[df['tourney_date'] <= config.data.train_cutoff].copy()
    mask_val = (df['tourney_date'] > config.data.train_cutoff) & (df['tourney_date'] < config.data.test_start)
    val = df[mask_val].copy()
    test = df[df['tourney_date'] >= config.data.test_start].copy()

    print(f"Data Splitting Complete: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test
