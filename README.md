# Trading Strategy — 5 Models (RF, XGB, CNN, LSTM, CNN-LSTM)

This project implements and backtests five models for predicting whether the H-th bar forward 
closes higher than the current bar close:

1. Random Forest (RF)
2. XGBoost (XGB)
3. Convolutional Neural Network (CNN)
4. Long Short-Term Memory Network (LSTM)
5. Hybrid CNN-LSTM

## Features
- Tick-level backtest with stop loss & trailing stop
- Feature importance plots (RF, XGB)
- Training history plots (CNN, LSTM, CNN-LSTM)
- Equity curve visualization
- Modular project structure for recruiters

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_main.py --parquet MBT_bars_5m_quant_PARIS_tz.parquet --H 10
```

## Outputs (in `outputs/`)
- `pred_{model}_H{H}.csv` – predicted probabilities
- `trade_log_{model}_H{H}.csv` – trade logs
- `equity_{model}_H{H}.png` – equity curves
- `feat_importance_{model}.png` – feature importances (RF/XGB)
- `history_{model}.png` – training histories (CNN/LSTM/Hybrid)
- `results_summary_H{H}.csv` – performance summary
