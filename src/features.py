import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class FeatureEngineer:
    def __init__(self, rolling_window: int = 10):
        self.window = rolling_window

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print(">>> Starting Feature Engineering...")
        df = df.copy()
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

        long_df = self._create_long_format(df)
        long_df = self._add_rolling_stats(long_df)
        long_df = self._add_h2h_features(long_df)
        long_df = self._add_days_since(long_df)
        final_df = self._pivot_to_match_format(long_df)
        final_df = self._add_diff_features(final_df)

        print(f">>> Feature Engineering Complete. Shape: {final_df.shape}")
        return final_df

    def _create_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'tourney_id' in df.columns:
            df['match_uid'] = df['tourney_id'].astype(str) + "_" + df['match_num'].astype(str)
        else:
            df['match_uid'] = df['tourney_date'].astype(str) + "_" + df['match_num'].astype(str)

        common_cols = ['tourney_date', 'surface', 'tourney_level', 'round', 'match_num', 'match_uid']
        actual_common = [c for c in common_cols if c in df.columns]

        w_cols = {'winner_name': 'player', 'winner_id': 'player_id', 'loser_name': 'opponent', 'loser_id': 'opponent_id', 'winner_rank': 'rank', 'loser_rank': 'opponent_rank', 'w_ace': 'ace', 'w_df': 'df', 'w_svpt': 'svpt', 'w_1stIn': '1stIn', 'w_1stWon': '1stWon', 'w_2ndWon': '2ndWon', 'w_bpSaved': 'bpSaved', 'w_bpFaced': 'bpFaced'}
        l_cols = {'loser_name': 'player', 'loser_id': 'player_id', 'winner_name': 'opponent', 'winner_id': 'opponent_id', 'loser_rank': 'rank', 'winner_rank': 'opponent_rank', 'l_ace': 'ace', 'l_df': 'df', 'l_svpt': 'svpt', 'l_1stIn': '1stIn', 'l_1stWon': '1stWon', 'l_2ndWon': '2ndWon', 'l_bpSaved': 'bpSaved', 'l_bpFaced': 'bpFaced'}

        df_w = df[actual_common + list(w_cols.keys())].rename(columns=w_cols).copy()
        df_w['label'] = 1
        df_l = df[actual_common + list(l_cols.keys())].rename(columns=l_cols).copy()
        df_l['label'] = 0

        long_df = pd.concat([df_w, df_l], axis=0)
        long_df = long_df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
        return long_df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon']
        df['1stIn_pct'] = df['1stIn'] / df['svpt'].replace(0, np.nan)
        df['win_pct'] = df['label']

        cols_to_roll = stats + ['1stIn_pct', 'win_pct']
        cols_to_roll = [c for c in cols_to_roll if c in df.columns]

        grouped = df.groupby('player')[cols_to_roll]

        # 1. ROLLING WINDOW (Trend)
        rolling_stats = grouped.apply(lambda x: x.shift(1).rolling(window=self.window, min_periods=1).mean())
        if isinstance(rolling_stats.index, pd.MultiIndex):
            rolling_stats = rolling_stats.reset_index(level=0, drop=True)
        rolling_stats = rolling_stats.sort_index()
        rolling_stats.columns = [f"{c}_roll" for c in rolling_stats.columns]

        # 2. LAG-1 FEATURES (Raw Last Match)
        # We grab the exact stats from the previous match (shift 1) without averaging
        lag_stats = grouped.apply(lambda x: x.shift(1))
        if isinstance(lag_stats.index, pd.MultiIndex):
            lag_stats = lag_stats.reset_index(level=0, drop=True)
        lag_stats = lag_stats.sort_index()
        lag_stats.columns = [f"{c}_lag" for c in lag_stats.columns]

        # Concat Original + Rolling + Lag
        return pd.concat([df, rolling_stats, lag_stats], axis=1)

    def _add_days_since(self, df: pd.DataFrame) -> pd.DataFrame:
        df['days_since'] = df.groupby('player')['tourney_date'].diff().dt.days
        df['days_since'] = df['days_since'].fillna(365)
        return df

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['win'] = df['label']
        h2h_grp = df.groupby(['player', 'opponent'])
        wins_series = h2h_grp['win'].apply(lambda x: x.shift(1).cumsum()).fillna(0)
        count_series = h2h_grp['win'].apply(lambda x: x.shift(1).expanding().count()).fillna(0)

        if isinstance(wins_series.index, pd.MultiIndex): wins_series = wins_series.droplevel([0, 1])
        if isinstance(count_series.index, pd.MultiIndex): count_series = count_series.droplevel([0, 1])

        df['h2h_wins'] = wins_series
        df['h2h_count'] = count_series
        df['h2h_win_rate'] = df['h2h_wins'] / df['h2h_count'].replace(0, np.nan)
        df['h2h_win_rate'] = df['h2h_win_rate'].fillna(0.5)
        return df.drop(columns=['win'])

    def _pivot_to_match_format(self, long_df: pd.DataFrame) -> pd.DataFrame:
        # Capture Rolling AND Lag columns
        roll_cols = [c for c in long_df.columns if '_roll' in c]
        lag_cols = [c for c in long_df.columns if '_lag' in c]

        meta_cols = ['match_uid', 'tourney_date', 'match_num', 'player', 'opponent', 'surface', 'tourney_level', 'round', 'label', 'rank', 'days_since', 'h2h_win_rate']

        # Raw stats (for LSTM history)
        raw_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon']

        cols_to_keep = meta_cols + roll_cols + lag_cols + raw_cols
        cols_to_keep = [c for c in cols_to_keep if c in long_df.columns]

        base = long_df[cols_to_keep].copy()

        opp_keep_cols = [c for c in cols_to_keep if c != 'label' and c != 'opponent']
        opp_stats = long_df[opp_keep_cols].copy()

        opp_stats = opp_stats.rename(columns={'player': 'opponent'})
        rename_map = {c: f"opponent_{c}" for c in opp_stats.columns if c not in ['match_uid', 'opponent']}
        opp_stats = opp_stats.rename(columns=rename_map)

        merged = pd.merge(base, opp_stats, on=['match_uid', 'opponent'], how='left')
        return merged

    def _add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Diff for Rolling
        features = ['ace_roll', 'df_roll', 'win_pct_roll', 'rank', '1stIn_pct_roll', 'svpt_roll']
        for f in features:
            if f in df.columns and f"opponent_{f}" in df.columns:
                df[f"{f}_diff"] = df[f] - df[f"opponent_{f}"]

        # Diff for Lag (Raw Last Match)
        lag_features = ['ace_lag', 'df_lag', 'win_pct_lag', '1stIn_pct_lag', 'svpt_lag']
        for f in lag_features:
            if f in df.columns and f"opponent_{f}" in df.columns:
                df[f"{f}_diff"] = df[f] - df[f"opponent_{f}"]

        return df
