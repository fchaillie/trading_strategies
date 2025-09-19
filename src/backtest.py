import numpy as np
import pandas as pd

def backtest_hold_extend_nextopen(
    df_5m: pd.DataFrame,
    df_ticks: pd.DataFrame,
    p_up: pd.Series,
    P_LONG: float,
    P_SHORT: float,
    contract_mult: float,
    commission_per_side: float,
    start_capital: float,
    stop_loss: float | None = None,
    trailing_stop: float | None = None
):
    cash = start_capital
    position = 0
    entry_price = None
    entry_time = None
    entry_prob = None
    bars_held = 0

    equity = []
    trades = []

    df_5m = df_5m.copy()
    df_5m["p_up"] = p_up.reindex(df_5m.index)

    for ts, bar in df_5m.iterrows():
        prob = bar["p_up"]

        # If in a trade, check tick-by-tick inside this 5m bar
        pnl_tick = 0.0
        if position != 0:
            bars_held += 1
            bar_start = ts
            bar_end = ts + pd.Timedelta(minutes=5)
            if len(df_ticks) > 0:
                ticks_in_bar = df_ticks.loc[bar_start:bar_end - pd.Timedelta(microseconds=1)]
            else:
                ticks_in_bar = pd.DataFrame()

            for tick_time, tick in ticks_in_bar.iterrows():
                price = float(tick["price"])
                pnl_tick = position * contract_mult * (price - entry_price)

                # Stop loss
                if stop_loss is not None and pnl_tick <= -stop_loss:
                    exit_price = price
                    trades[-1].update({
                        "exit_time": tick_time,
                        "exit_price": exit_price,
                        "bars_held": bars_held,
                        "pnl_gross": pnl_tick,
                        "commission_usd": float(commission_per_side * 2.0),
                        "pnl_net": float(pnl_tick - float(commission_per_side * 2.0)),
                        "exit_reason": "stop_loss"
                    })
                    cash += pnl_tick - commission_per_side
                    position = 0
                    entry_price = entry_time = entry_prob = None
                    bars_held = 0
                    break

                # Trailing stop
                if trailing_stop is not None and len(trades) > 0:
                    if "max_fav" not in trades[-1]:
                        trades[-1]["max_fav"] = pnl_tick
                    else:
                        trades[-1]["max_fav"] = max(trades[-1]["max_fav"], pnl_tick)
                    trail_level = trades[-1]["max_fav"] - trailing_stop
                    if pnl_tick <= trail_level:
                        exit_price = price
                        trades[-1].update({
                            "exit_time": tick_time,
                            "exit_price": exit_price,
                            "bars_held": bars_held,
                            "pnl_gross": pnl_tick,
                            "commission_usd": float(commission_per_side * 2.0),
                            "pnl_net": float(pnl_tick - float(commission_per_side * 2.0)),
                            "exit_reason": "trailing_stop"
                        })
                        cash += pnl_tick - commission_per_side
                        position = 0
                        entry_price = entry_time = entry_prob = None
                        bars_held = 0
                        break

        # Entry decision at the close of 5m bar
        if position == 0 and pd.notna(prob):
            if prob >= P_LONG:
                position = 1
            elif prob <= P_SHORT:
                position = -1

            if position != 0:
                entry_price = bar["close"]
                entry_time  = ts + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
                entry_prob  = prob
                cash       -= commission_per_side

                trades.append({
                    "entry_time": entry_time,
                    "entry_price": float(entry_price),
                    "entry_prob": float(round(float(prob), 4)),
                    "side": "LONG" if position == 1 else "SHORT",
                    "side_mult": int(position),
                    "lot_size": float(contract_mult)
                })

        if position != 0:
            pnl_tick = position * contract_mult * (bar["close"] - entry_price)
            eq = cash + pnl_tick
        else:
            eq = cash
        equity.append(eq)

    equity = pd.Series(equity, index=df_5m.index, name="equity")
    trades = pd.DataFrame(trades)
    return equity, trades
