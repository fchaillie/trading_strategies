# === CONFIGURATION ===

# Data paths (update when running)
PARQUET_PATH = "MBT_bars_5m_quant_PARIS_tz.parquet"
TICKS_PATH   = "ticks.parquet"

# Trading params
CONTRACT_MULTIPLIER = 0.1
COMMISSION_PER_SIDE = 0.5
START_CAPITAL = 10_000.0
H = 10

# Prob thresholds
P_LONG = 0.5
P_SHORT = 0.2

# Risk management
STOP_LOSS = 100     # in USD (per contract)
TRAILING_STOP = 100 # in USD (per contract)

# Randomness
SEED = 42

# Random Forest / XGB params
N_ESTIMATORS = 350
MAX_DEPTH = None
N_JOBS = -1

# Neural net params
EPOCHS = 10
BATCH_SIZE = 32
WINDOW_SIZE = 20   # for CNN/LSTM sequence length
