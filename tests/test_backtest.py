import pandas as pd
from src.backtest import backtest_hold_extend_nextopen

def test_backtest_runs():
    ts = pd.date_range("2024-01-01", periods=20, freq="5min")
    df_5m = pd.DataFrame({
        "open": range(20),
        "high": [x+1 for x in range(20)],
        "low":  [x-1 for x in range(20)],
        "close": [x+0.5 for x in range(20)],
    }, index=ts)
    ticks = pd.DataFrame({"price": range(100)}, index=pd.date_range("2024-01-01", periods=100, freq="T"))
    p_up = pd.Series(0.5, index=ts)
    eq, trades = backtest_hold_extend_nextopen(
        df_5m, ticks, p_up,
        P_LONG=0.6, P_SHORT=0.4,
        contract_mult=0.1, commission_per_side=0.5, start_capital=10000.0,
        stop_loss=100, trailing_stop=100
    )
    assert isinstance(eq, pd.Series)
    assert isinstance(trades, pd.DataFrame)
