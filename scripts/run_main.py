#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from src.config import (
    PARQUET_PATH, CONTRACT_MULTIPLIER, COMMISSION_PER_SIDE, START_CAPITAL,
    STOP_LOSS, TRAILING_STOP, P_LONG, P_SHORT
)
from src.config import cols_to_use
from src.data_utils import build_dataset, mask_bars
from src.model_rf import fit_predict_block_rf
from src.backtest import backtest_hold_extend_nextopen
from src.metrics import max_drawdown

def load_ticks_or_synthesize(df_5m: pd.DataFrame, ticks_parquet: str | None):
    if ticks_parquet:
        return pd.read_parquet(ticks_parquet).sort_index()
    # synthesize naive ticks: 60 ticks per 5m bar, linearly between open->close
    rows = []
    for ts, row in df_5m.iterrows():
        start = ts
        end = ts + pd.Timedelta(minutes=5)
        prices = np.linspace(row['open'], row['close'], 60)
        idx = pd.date_range(start, periods=60, end=end, inclusive="left")
        rows.append(pd.DataFrame({"price": prices}, index=idx))
    return pd.concat(rows).sort_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=PARQUET_PATH)
    ap.add_argument("--ticks_parquet", default=None, help="Optional ticks parquet (index dtype datetime, col 'price')")
    ap.add_argument("--h_start", type=int, default=21)
    ap.add_argument("--h_end", type=int, default=27)
    ap.add_argument("--p_long", type=float, default=P_LONG)
    ap.add_argument("--p_short", type=float, default=P_SHORT)
    args = ap.parse_args()

    results = []
    P_LONG_ = args.p_long
    P_SHORT_ = args.p_short

    for H in range(args.h_start, args.h_end):  # inclusive of start, exclusive of end
        print(f"\n=== Running H={H} ===")
        df = build_dataset(args.parquet, H)

        n = len(df)
        split_idx = int(n * 0.8)
        train_idx = df.index[:split_idx]
        test_idx  = df.index[split_idx:]

        df.loc[train_idx, "y"] = mask_bars(df.loc[train_idx, "y"].copy(), 30, H, 30, H)
        df.loc[test_idx, "y"] = mask_bars(df.loc[test_idx, "y"].copy(), 30, H, 30, H)

        p_up_test, model, scaler = fit_predict_block_rf(df, train_idx, test_idx, feat_cols=cols_to_use)
        p_up_test = mask_bars(p_up_test, 30, H, 30, H)

        df_test = df.loc[test_idx]
        y_test  = df_test["y"]

        # Load ticks (or synthesize)
        start_test = df_test.index.min()
        end_test = df_test.index.max()
        df_ticks = load_ticks_or_synthesize(df_test[["open","high","low","close"]], args.ticks_parquet)

        print("Bars:", df_test.index.min(), "to", df_test.index.max())
        if len(df_ticks):
            print("Ticks:", df_ticks.index.min(), "to", df_ticks.index.max())

        equity_test, trades_test = backtest_hold_extend_nextopen(
            df_5m = df_test,
            df_ticks = df_ticks,
            p_up = p_up_test,
            P_LONG = P_LONG_,
            P_SHORT = P_SHORT_,
            contract_mult = CONTRACT_MULTIPLIER,
            commission_per_side = COMMISSION_PER_SIDE,
            start_capital = START_CAPITAL,
            stop_loss = STOP_LOSS,
            trailing_stop = TRAILING_STOP
        )

        total_commissions = trades_test.get("commission_usd", pd.Series(dtype=float)).sum() if len(trades_test) else 0.0
        total_pnl_net = trades_test.get("pnl_net", pd.Series(dtype=float)).sum() if len(trades_test) else 0.0

        y_pred_test = (p_up_test >= P_LONG_).astype(float)

        mask_valid = y_test.notna() & y_pred_test.notna()
        if mask_valid.any():
            y_true = y_test[mask_valid].astype(int)
            y_pred = y_pred_test[mask_valid].astype(int)
            precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        else:
            precision = [np.nan, np.nan]
            recall    = [np.nan, np.nan]

        dd_test = max_drawdown(equity_test)

        outdir = "outputs"
        import os
        os.makedirs(outdir, exist_ok=True)
        trades_test.to_csv(f"{outdir}/trade_log_H{H}_LONG{P_LONG_}_SHORT{P_SHORT_}.csv", index=False)

        # Plot equity curve
        plt.figure(figsize=(7,3))
        plt.plot(equity_test.index, equity_test, label="Equity")
        plt.title(f"Equity Curve H={H} LONG={P_LONG_} SHORT={P_SHORT_}")
        plt.xlabel("Time"); plt.ylabel("Portfolio ($)"); plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/equity_H{H}_L{P_LONG_}_S{P_SHORT_}.png", dpi=140)
        plt.close()

        results.append({
            "H": H,
            "P_LONG": round(P_LONG_, 2),
            "P_SHORT": round(P_SHORT_, 2),
            "Trades_Test": int(len(trades_test)),
            "Test_equity": float(equity_test.iloc[-1]) if len(equity_test) else float("nan"),
            "Total_com": float(total_commissions),
            "Total_Net_PNL": float(total_pnl_net),
            "MaxDD_Test": float(dd_test),
            "Precision_Down": float(precision[0]) if precision[0]==precision[0] else None,
            "Recall_Down": float(recall[0]) if recall[0]==recall[0] else None,
            "Precision_Up": float(precision[1]) if precision[1]==precision[1] else None,
            "Recall_Up": float(recall[1]) if recall[1]==recall[1] else None
        })

    results_df = pd.DataFrame(results)
    print("\n=== Summary of Results ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(results_df)
    results_df.to_csv("outputs/summary_results.csv", index=False)

if __name__ == "__main__":
    main()
