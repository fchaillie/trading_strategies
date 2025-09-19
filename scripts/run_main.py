import argparse
import os
import pandas as pd
import numpy as np

from src import config
from src.data_utils import build_dataset, mask_bars, create_sequences
from src.backtest import backtest_hold_extend_nextopen
from src.models import (
    fit_predict_block_rf,
    fit_predict_block_xgb,
    fit_predict_block_cnn,
    fit_predict_block_lstm,
    fit_predict_block_cnnlstm,
)
from src.metrics import compute_classification_metrics, max_drawdown
from src.visualization import plot_equity_curve, plot_feature_importance, plot_training_history

def main(parquet_path, ticks_path, H, P_LONG, P_SHORT):
    os.makedirs("outputs", exist_ok=True)

    # === Load dataset ===
    df = build_dataset(parquet_path, H)
    n = len(df)
    split_idx = int(n * 0.8)
    train_idx = df.index[:split_idx]
    test_idx  = df.index[split_idx:]

    y = df["y"].astype(int)
    X = df.drop(columns=["y"])

    # Train/test split
    X_train, y_train = X.loc[train_idx].values, y.loc[train_idx].values
    X_test, y_test   = X.loc[test_idx].values, y.loc[test_idx].values

    results = []

    # Helper to backtest and log results
    def evaluate_model(name, prob_up, model=None, history=None):
        df_test = df.loc[test_idx].copy()
        p_up_test = pd.Series(prob_up, index=test_idx, name="p_up")
        df_test["p_up"] = p_up_test

        # Load ticks subset
        if os.path.exists(ticks_path):
            df_ticks = pd.read_parquet(ticks_path)
            df_ticks_test = df_ticks.loc[df_test.index.min():df_test.index.max()]
        else:
            # synthetic ticks if no tick file
            df_ticks_test = pd.DataFrame(index=df_test.index, columns=["price"])
            df_ticks_test["price"] = df_test["close"]

        # Backtest
        equity, trades = backtest_hold_extend_nextopen(
            df_5m=df_test,
            df_ticks=df_ticks_test,
            p_up=p_up_test,
            P_LONG=P_LONG,
            P_SHORT=P_SHORT,
            contract_mult=config.CONTRACT_MULTIPLIER,
            commission_per_side=config.COMMISSION_PER_SIDE,
            start_capital=config.START_CAPITAL,
            stop_loss=config.STOP_LOSS,
            trailing_stop=config.TRAILING_STOP,
        )

        # Metrics
        y_pred = (p_up_test >= P_LONG).astype(int)
        mask_valid = y_test.notna()
        metrics = compute_classification_metrics(y_test[mask_valid], y_pred[mask_valid])
        dd = max_drawdown(equity)
        total_com = trades["commission_usd"].sum() if not trades.empty else 0.0
        total_pnl_net = trades["pnl_net"].sum() if not trades.empty else 0.0

        # Save outputs
        trades.to_csv(f"outputs/trade_log_{name}_H{H}.csv", index=False)
        equity.to_csv(f"outputs/pred_{name}_H{H}.csv")
        plot_equity_curve(equity, name, H, P_LONG, P_SHORT, save_path=f"outputs/equity_{name}_H{H}.png")

        if model is not None and hasattr(model, "feature_importances_"):
            plot_feature_importance(model, list(X.columns), save_path=f"outputs/feat_importance_{name}.png")

        if history is not None:
            plot_training_history(history, name, save_path=f"outputs/history_{name}.png")

        results.append({
            "Model": name,
            "H": H,
            "P_LONG": P_LONG,
            "P_SHORT": P_SHORT,
            "Trades_Test": len(trades),
            "Test_equity": equity.iloc[-1],
            "Total_com": total_com,
            "Total_Net_PNL": total_pnl_net,
            "MaxDD_Test": dd,
            **metrics
        })

    # === RF ===
    prob_up, model, scaler = fit_predict_block_rf(X_train, y_train, X_test)
    evaluate_model("rf", prob_up, model=model)

    # === XGB ===
    prob_up, model, scaler = fit_predict_block_xgb(X_train, y_train, X_test)
    evaluate_model("xgb", prob_up, model=model)

    # === CNN ===
    Xs_train, ys_train = create_sequences(X_train, y_train, config.WINDOW_SIZE)
    Xs_test, ys_test   = create_sequences(X_test, y_test, config.WINDOW_SIZE)
    prob_up, model, history = fit_predict_block_cnn(Xs_train, ys_train, Xs_test, epochs=config.EPOCHS)
    evaluate_model("cnn", prob_up, model=model, history=history)

    # === LSTM ===
    prob_up, model, history = fit_predict_block_lstm(Xs_train, ys_train, Xs_test, epochs=config.EPOCHS)
    evaluate_model("lstm", prob_up, model=model, history=history)

    # === CNN-LSTM ===
    prob_up, model, history = fit_predict_block_cnnlstm(Xs_train, ys_train, Xs_test, epochs=config.EPOCHS)
    evaluate_model("cnnlstm", prob_up, model=model, history=history)

    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"outputs/results_summary_H{H}.csv", index=False)
    print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=config.PARQUET_PATH, help="Path to OHLCV parquet file")
    parser.add_argument("--ticks_parquet", type=str, default=config.TICKS_PATH, help="Path to tick parquet file")
    parser.add_argument("--H", type=int, default=config.H, help="Prediction horizon")
    parser.add_argument("--p_long", type=float, default=config.P_LONG, help="Prob threshold for long")
    parser.add_argument("--p_short", type=float, default=config.P_SHORT, help="Prob threshold for short")
    args = parser.parse_args()

    main(args.parquet, args.ticks_parquet, args.H, args.p_long, args.p_short)
