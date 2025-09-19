# ---------------- CONFIG ----------------
PARQUET_PATH = "MBT_bars_5m_quant_PARIS_tz.parquet"

CONTRACT_MULTIPLIER = 0.1
COMMISSION_PER_SIDE = 0.5
START_CAPITAL = 10_000.0

# Default horizon
H = 10

P_LONG = 0.5
P_SHORT = 0.2

# --- Risk management params ---
STOP_LOSS = 100     # USD per contract
TRAILING_STOP = 100 # USD per contract

SEED = 42

N_ESTIMATORS = 350
MAX_DEPTH = None
N_JOBS = -1

# Feature columns
cols_to_use = [
    'volume','trade_count','rsi_14','macd_hist','bb_width_20_2',
    'parkinson_var_5','gk_var_5','parkinson_var_15','gk_var_15',
    'parkinson_var_30','gk_var_30','z_volume_20','z_range_20','z_ret_20',
    'session_progress','rv_5','rv_15','rv_30','notional','cum_notional',
    'cum_volume','open','high','low','close'
]
