import numpy as np
import pandas as pd
from typing import List, Tuple
from pandas.tseries.offsets import DateOffset
from .config import cols_to_use

def build_dataset(path: str, H: int) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=cols_to_use).sort_index()
    df["close_fut"] = df["close"].shift(-H)
    df["y"] = (df["close_fut"] > df["close"]).astype(int)
    df = df.drop(columns=["close_fut"])
    df = df.dropna(subset=["y"])
    return df

def rolling_walkforward_indices(df: pd.DataFrame,
                                start: pd.Timestamp,
                                end: pd.Timestamp,
                                train_len: pd.DateOffset = DateOffset(months=18),
                                test_len: pd.DateOffset = DateOffset(months=12),
                                step: pd.DateOffset = DateOffset(months=8)
                                ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    pairs = []
    cur_start = start
    while True:
        train_start = cur_start
        train_end   = train_start + train_len
        test_end    = train_end + test_len
        if test_end > end:
            break
        tr_idx = df[(df.index >= train_start) & (df.index < train_end)].index
        te_idx = df[(df.index >= train_end) & (df.index < test_end)].index
        if len(tr_idx) > 100 and len(te_idx) > 0:
            pairs.append((tr_idx, te_idx))
        cur_start += step
    return pairs

def mask_bars(
    prob_series: pd.Series,
    n_bars_start_day: int = 30,
    n_bars_end_day: int = 5,
    n_bars_start_total: int = 30,
    n_bars_end_total: int = 5
) -> pd.Series:
    df = prob_series.to_frame("p_up")
    df["date"] = df.index.date

    if n_bars_start_day > 0:
        first_mask_day = df.groupby("date").head(n_bars_start_day).index
        prob_series.loc[first_mask_day] = np.nan

    if n_bars_end_day > 0:
        last_mask_day = df.groupby("date").tail(n_bars_end_day).index
        prob_series.loc[last_mask_day] = np.nan

    if n_bars_start_total > 0:
        start_mask_total = prob_series.index[:n_bars_start_total]
        prob_series.loc[start_mask_total] = np.nan

    if n_bars_end_total > 0:
        end_mask_total = prob_series.index[-n_bars_end_total:]
        prob_series.loc[end_mask_total] = np.nan

    return prob_series
