import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

# === DATA PREP ===
def build_dataset(path: str, H: int, cols_to_use=None) -> pd.DataFrame:
    if cols_to_use is None:
        cols_to_use = ['volume', 'trade_count','rsi_14', 'macd_hist', 'bb_width_20_2',
                       'parkinson_var_5', 'gk_var_5', 'parkinson_var_15', 'gk_var_15',
                       'parkinson_var_30', 'gk_var_30','z_volume_20', 'z_range_20', 'z_ret_20',
                       'session_progress', 'rv_5', 'rv_15', 'rv_30', 'notional', 'cum_notional',
                       'cum_volume','open', 'high', 'low', 'close']
    df = pd.read_parquet(path, columns=cols_to_use).sort_index()
    df["close_fut"] = df["close"].shift(-H)
    df["y"] = (df["close_fut"] > df["close"]).astype(int)
    df = df.drop(columns=["close_fut"])
    df = df.dropna(subset=["y"])
    return df

def mask_bars(prob_series: pd.Series,
              n_bars_start_day: int = 30,
              n_bars_end_day: int = 5,
              n_bars_start_total: int = 30,
              n_bars_end_total: int = 5) -> pd.Series:
    df = prob_series.to_frame("p_up")
    df["date"] = df.index.date

    # Mask start-of-day
    if n_bars_start_day > 0:
        first_mask_day = df.groupby("date").head(n_bars_start_day).index
        prob_series.loc[first_mask_day] = np.nan

    # Mask end-of-day
    if n_bars_end_day > 0:
        last_mask_day = df.groupby("date").tail(n_bars_end_day).index
        prob_series.loc[last_mask_day] = np.nan

    # Mask start of entire series
    if n_bars_start_total > 0:
        start_mask_total = prob_series.index[:n_bars_start_total]
        prob_series.loc[start_mask_total] = np.nan

    # Mask end of entire series
    if n_bars_end_total > 0:
        end_mask_total = prob_series.index[-n_bars_end_total:]
        prob_series.loc[end_mask_total] = np.nan

    return prob_series

def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)
