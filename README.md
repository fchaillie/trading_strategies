# Trading Strategy — 5 Models (RF, XGB, CNN, LSTM, CNN-LSTM)

This project demonstrates machine learning for financial time series prediction.  
We forecast whether the H-th bar forward closes higher than the current bar, and backtest the strategy at tick level with stop-loss and trailing-stop risk management.

We evaluate five models side by side:

- 🟢 **Random Forest (RF)** – classic ensemble baseline  
- 🟠 **XGBoost (XGB)** – gradient boosting for nonlinear patterns  
- 🔵 **CNN** – local feature extraction from rolling windows  
- 🟣 **LSTM** – sequence model for temporal dependencies  
- ⚡ **CNN-LSTM Hybrid** – pattern detection + sequence memory  

---

## 🚀 Features
- Tick-level backtest with stop-loss & trailing stop  
- Feature importance plots (RF, XGB)  
- Training history plots (CNN, LSTM, Hybrid)  
- Equity curve visualization  
- Modular project structure recruiters will appreciate  

---

## ⚙️ Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training + backtest (80/20 split)
python scripts/run_main.py --parquet MBT_bars_5m_quant_PARIS_tz.parquet --H 10
```

Outputs are saved in the `outputs/` folder:

- Trade logs (`trade_log_{model}_H{H}.csv`)  
- Probabilities (`pred_{model}_H{H}.csv`)  
- Equity curves (`equity_{model}_H{H}.png`)  
- Feature importances (`feat_importance_{model}.png`)  
- Training histories (`history_{model}.png`)  
- Summary (`results_summary_H{H}.csv`)  

---

## 📂 Project Structure

```
trading_strategies/
├── data/                # sample parquet/csv
├── scripts/
│   ├── run_main.py      # main entry point
│   ├── data_prep.py
│   ├── models.py
│   ├── backtest.py
│   ├── metrics.py
│   └── visualize.py
├── outputs/             # saved results
├── docs/img/            # images for README
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Results by Model

### 1. Random Forest (RF)
- Captures nonlinear interactions between features.  
- Provides feature importances for interpretability.  
- Baseline performance for comparison.  

📈 Equity Curve RF  
![Equity Curve RF](docs/img/equity_rf.png)  

📊 Feature Importance RF  
![Feature Importance RF](docs/img/feat_importance_rf.png)  

---

### 2. XGBoost (XGB)
- Strong gradient boosting method for tabular financial data.  
- Often outperforms Random Forest with proper tuning.  

📈 Equity Curve XGB  
![Equity Curve XGB](docs/img/equity_xgb.png)  

📊 Feature Importance XGB  
![Feature Importance XGB](docs/img/feat_importance_xgb.png)  

---

### 3. Convolutional Neural Network (CNN)
- Learns local patterns from sliding windows of OHLCV + features.  
- Good for detecting micro-structures or breakouts.  

📈 Equity Curve CNN  
![Equity Curve CNN](docs/img/equity_cnn.png)  

📉 Training History CNN  
![Training History CNN](docs/img/history_cnn.png)  

---

### 4. Long Short-Term Memory Network (LSTM)
- Sequence model capturing temporal dependencies.  
- Useful for trend continuation or reversal detection.  

📈 Equity Curve LSTM  
![Equity Curve LSTM](docs/img/equity_lstm.png)  

📉 Training History LSTM  
![Training History LSTM](docs/img/history_lstm.png)  

---

### 5. CNN-LSTM Hybrid
- Combines CNN (local feature extraction) + LSTM (sequence memory).  
- Often the strongest performer on sequential trading data.  

📈 Equity Curve CNN-LSTM  
![Equity Curve CNN-LSTM](docs/img/equity_cnn_lstm.png)  

📉 Training History CNN-LSTM  
![Training History CNN-LSTM](docs/img/history_cnn_lstm.png)  

---

## 📋 Results Summary Table

| Model      | Accuracy | ROC AUC | Sharpe | Max DD | CAGR |
|------------|----------|---------|--------|--------|------|
| RF         |  ...     |   ...   |  ...   |  ...   | ...  |
| XGB        |  ...     |   ...   |  ...   |  ...   | ...  |
| CNN        |  ...     |   ...   |  ...   |  ...   | ...  |
| LSTM       |  ...     |   ...   |  ...   |  ...   | ...  |
| CNN-LSTM   |  ...     |   ...   |  ...   |  ...   | ...  |

---

## 🎯 Why this project

This project is designed as a portfolio piece to showcase:

- Financial data handling (OHLCV + tick data)  
- Application of both classic ML (RF/XGB) and deep learning (CNN/LSTM)  
- Proper backtesting with risk management  
- Clean, modular, recruiter-friendly codebase  

---

## 📜 License

MIT License — free to use and adapt.  

---

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
