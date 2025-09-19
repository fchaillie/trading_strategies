
# Trading Strategy — 5 Models (RF, XGB, CNN, LSTM, CNN-LSTM)

This project demonstrates **machine learning for financial time series prediction**.  
We forecast whether the **H-th bar forward closes higher** than the current bar, and backtest the strategy at **tick level** with stop-loss and trailing-stop risk management.

The repo showcases **5 models**:

1. 🟢 **Random Forest (RF)** – classic ensemble baseline  
2. 🟠 **XGBoost (XGB)** – gradient boosting for nonlinear patterns  
3. 🔵 **CNN** – local feature extraction from rolling windows  
4. 🟣 **LSTM** – sequence model for temporal dependencies  
5. ⚡ **CNN-LSTM Hybrid** – pattern detection + sequence memory  

---

## 🚀 Features
- **Tick-level backtest** with stop-loss & trailing stop  
- **Feature importance plots** (RF, XGB)  
- **Training history plots** (CNN, LSTM, Hybrid)  
- **Equity curve visualization**  
- **Modular project structure** recruiters will appreciate  

---

## 📂 Project Structure
```
trading-strategy-5models/
├── src/
│   ├── config.py          # default params (STOP_LOSS, TRAILING_STOP, etc.)
│   ├── data_utils.py      # build dataset, mask bars, sequence creation
│   ├── backtest.py        # tick-level backtest logic
│   ├── models.py          # RF, XGB, CNN, LSTM, CNN-LSTM
│   ├── metrics.py         # classification metrics & max drawdown
│   └── visualization.py   # plotting utilities
├── scripts/
│   └── run_main.py        # trains & tests all 5 models
├── README.md
├── requirements.txt
└── .gitignore
```

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

---

## 📊 Example Outputs

When you run the pipeline, you’ll find results in `outputs/`:

- **Equity Curves**  
  ![Equity Curve](outputs/equity_rf_H10.png)

- **Feature Importances (RF/XGB)**  
  ![Feature Importance](outputs/feat_importance_rf.png)

- **Training History (CNN/LSTM/Hybrid)**  
  ![Training History](outputs/history_lstm.png)

- **Performance Summary**  
  `results_summary_H10.csv`  

---

## 🎯 Why this project
This project is designed as a **portfolio piece** to showcase:
- Financial data handling (OHLCV + tick data)  
- Application of both **classic ML** (RF/XGB) and **deep learning** (CNN/LSTM)  
- Proper backtesting with **risk management**  
- Clean, modular, recruiter-friendly codebase  

---

## 📜 License
MIT License — free to use and adapt.
