import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any, Self

from src.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, rolling_window: int = 10) -> None:
        self.window: int = rolling_window
        self.preprocessor: Any = None

    @classmethod
    def load_state(cls, base_path: Path = Path("artifacts/prod")) -> Self:
        """
        Loads the feature engineer and the production stateful preprocessor.
        """
        instance = cls()
        preprocessor_path = base_path / "global_preprocessor.pkl"
        try:
            instance.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
        except FileNotFoundError as e:
            logger.error(f"Preprocessor not found at {preprocessor_path}.")
            raise FileNotFoundError("Missing global_preprocessor.pkl in artifacts/prod/") from e
        return instance

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(">>> Starting Feature Engineering...")
        df_copy: pd.DataFrame = df.copy()
        df_copy['tourney_date'] = pd.to_datetime(df_copy['tourney_date'], format='%Y%m%d', errors='coerce')
        df_sorted: pd.DataFrame = df_copy.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

        long_df: pd.DataFrame = self._create_long_format(df_sorted)
        long_df = self._add_rolling_stats(long_df)
        long_df = self._add_h2h_features(long_df)
        long_df = self._add_days_since(long_df)
        final_df: pd.DataFrame = self._pivot_to_match_format(long_df)
        final_df = self._add_diff_features(final_df)

        logger.info(f">>> Feature Engineering Complete. Shape: {final_df.shape}")
        return final_df

    def _create_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        if 'tourney_id' in df_copy.columns:
            df_copy['match_uid'] = df_copy['tourney_id'].astype(str) + "_" + df_copy['match_num'].astype(str)
        else:
            df_copy['match_uid'] = df_copy['tourney_date'].astype(str) + "_" + df_copy['match_num'].astype(str)

        common_cols: List[str] = ['tourney_date', 'surface', 'tourney_level', 'round', 'match_num', 'match_uid']
        
        if 'is_inference' in df_copy.columns:
            common_cols.append('is_inference')
            
        actual_common: List[str] = [c for c in common_cols if c in df_copy.columns]

        w_cols: Dict[str, str] = {'winner_name': 'player', 'winner_id': 'player_id', 'loser_name': 'opponent', 'loser_id': 'opponent_id',
                                  'winner_rank': 'rank', 'loser_rank': 'opponent_rank', 'w_ace': 'ace', 'w_df': 'df', 'w_svpt': 'svpt',
                                  'w_1stIn': '1stIn', 'w_1stWon': '1stWon', 'w_2ndWon': '2ndWon', 'w_bpSaved': 'bpSaved', 'w_bpFaced': 'bpFaced'}
        l_cols: Dict[str, str] = {'loser_name': 'player', 'loser_id': 'player_id', 'winner_name': 'opponent', 'winner_id': 'opponent_id',
                                  'loser_rank': 'rank', 'winner_rank': 'opponent_rank', 'l_ace': 'ace', 'l_df': 'df', 'l_svpt': 'svpt',
                                  'l_1stIn': '1stIn', 'l_1stWon': '1stWon', 'l_2ndWon': '2ndWon', 'l_bpSaved': 'bpSaved', 'l_bpFaced': 'bpFaced'}

        df_w: pd.DataFrame = df_copy[actual_common + list(w_cols.keys())].rename(columns=w_cols).copy()
        df_w['label'] = 1
        df_l: pd.DataFrame = df_copy[actual_common + list(l_cols.keys())].rename(columns=l_cols).copy()
        df_l['label'] = 0

        long_df: pd.DataFrame = pd.concat([df_w, df_l], axis=0)
        long_df_sorted: pd.DataFrame = long_df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
        return long_df_sorted

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        stats: List[str] = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon']
        df_copy['1stIn_pct'] = df_copy['1stIn'] / df_copy['svpt'].replace(0, np.nan)
        df_copy['win_pct'] = df_copy['label']

        cols_to_roll: List[str] = stats + ['1stIn_pct', 'win_pct']
        cols_to_roll = [c for c in cols_to_roll if c in df_copy.columns]

        grouped: pd.core.groupby.generic.DataFrameGroupBy = df_copy.groupby('player')[cols_to_roll]

        rolling_stats: pd.DataFrame = grouped.apply(lambda x: x.shift(1).rolling(window=self.window, min_periods=1).mean())
        if isinstance(rolling_stats.index, pd.MultiIndex):
            rolling_stats = rolling_stats.reset_index(level=0, drop=True)
        rolling_stats = rolling_stats.sort_index()
        rolling_stats.columns = [f"{c}_roll" for c in rolling_stats.columns]

        lag_stats: pd.DataFrame = grouped.apply(lambda x: x.shift(1))
        if isinstance(lag_stats.index, pd.MultiIndex):
            lag_stats = lag_stats.reset_index(level=0, drop=True)
        lag_stats = lag_stats.sort_index()
        lag_stats.columns = [f"{c}_lag" for c in lag_stats.columns]

        return pd.concat([df_copy, rolling_stats, lag_stats], axis=1)

    def _add_days_since(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        df_copy['days_since'] = df_copy.groupby('player')['tourney_date'].diff().dt.days
        df_copy['days_since'] = df_copy['days_since'].fillna(365)
        return df_copy

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        df_copy['win'] = df_copy['label']
        h2h_grp: pd.core.groupby.generic.DataFrameGroupBy = df_copy.groupby(['player', 'opponent'])
        wins_series: pd.Series = h2h_grp['win'].apply(lambda x: x.shift(1).cumsum()).fillna(0)
        count_series: pd.Series = h2h_grp['win'].apply(lambda x: x.shift(1).expanding().count()).fillna(0)

        if isinstance(wins_series.index, pd.MultiIndex): wins_series = wins_series.droplevel([0, 1])
        if isinstance(count_series.index, pd.MultiIndex): count_series = count_series.droplevel([0, 1])

        df_copy['h2h_wins'] = wins_series
        df_copy['h2h_count'] = count_series
        df_copy['h2h_win_rate'] = df_copy['h2h_wins'] / df_copy['h2h_count'].replace(0, np.nan)
        df_copy['h2h_win_rate'] = df_copy['h2h_win_rate'].fillna(0.5)
        return df_copy.drop(columns=['win'])

    def _pivot_to_match_format(self, long_df: pd.DataFrame) -> pd.DataFrame:
        roll_cols: List[str] = [c for c in long_df.columns if '_roll' in c]
        lag_cols: List[str] = [c for c in long_df.columns if '_lag' in c]

        meta_cols: List[str] = ['match_uid', 'tourney_date', 'match_num', 'player', 'opponent', 'surface', 'tourney_level', 'round', 'label', 'rank', 'days_since', 'h2h_win_rate']
        
        if 'is_inference' in long_df.columns:
            meta_cols.append('is_inference')

        raw_cols: List[str] = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon']

        cols_to_keep: List[str] = meta_cols + roll_cols + lag_cols + raw_cols
        cols_to_keep = [c for c in cols_to_keep if c in long_df.columns]

        base: pd.DataFrame = long_df[cols_to_keep].copy()

        opp_keep_cols: List[str] = [c for c in cols_to_keep if c not in ['label', 'opponent', 'is_inference']]
        opp_stats: pd.DataFrame = long_df[opp_keep_cols].copy()

        opp_stats = opp_stats.rename(columns={'player': 'opponent'})
        rename_map: Dict[str, str] = {c: f"opponent_{c}" for c in opp_stats.columns if c not in ['match_uid', 'opponent']}
        opp_stats = opp_stats.rename(columns=rename_map)

        merged: pd.DataFrame = pd.merge(base, opp_stats, on=['match_uid', 'opponent'], how='left')
        return merged

    def _add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        features: List[str] = ['ace_roll', 'df_roll', 'win_pct_roll', 'rank', '1stIn_pct_roll', 'svpt_roll']
        for f in features:
            if f in df_copy.columns and f"opponent_{f}" in df_copy.columns:
                df_copy[f"{f}_diff"] = df_copy[f] - df_copy[f"opponent_{f}"]

        lag_features: List[str] = ['ace_lag', 'df_lag', 'win_pct_lag', '1stIn_pct_lag', 'svpt_lag']
        for f in lag_features:
            if f in df_copy.columns and f"opponent_{f}" in df_copy.columns:
                df_copy[f"{f}_diff"] = df_copy[f] - df_copy[f"opponent_{f}"]

        return df_copy