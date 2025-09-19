
# Trading Strategy — 5 Models (RF, XGB, CNN, LSTM, CNN-LSTM)

This project demonstrates **machine learning for financial time series prediction**.  
We forecast whether the **H-th bar forward closes higher** than the current bar, and backtest the strategy at **tick level** with stop-loss and trailing-stop risk management.

We evaluate **five models** side by side:

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

## 📊 Results by Model

### 1. Random Forest (RF)
- Captures nonlinear interactions between features.
- Provides **feature importances** for interpretability.
- Baseline performance for comparison.

📈 *Equity Curve RF*  
👉 *(Insert screenshot here)*

📊 *Feature Importance RF*  
👉 *(Insert screenshot here)*

---

### 2. XGBoost (XGB)
- Strong gradient boosting method for tabular financial data.
- Often outperforms Random Forest with proper tuning.

📈 *Equity Curve XGB*  
👉 *(Insert screenshot here)*

📊 *Feature Importance XGB*  
👉 *(Insert screenshot here)*

---

### 3. Convolutional Neural Network (CNN)
- Learns **local patterns** from sliding windows of OHLCV + features.
- Good for detecting micro-structures or breakouts.

📈 *Equity Curve CNN*  
👉 *(Insert screenshot here)*

📉 *Training History CNN*  
👉 *(Insert screenshot here)*

---

### 4. Long Short-Term Memory Network (LSTM)
- Sequence model capturing **temporal dependencies**.
- Useful for trend continuation or reversal detection.

📈 *Equity Curve LSTM*  
👉 *(Insert screenshot here)*

📉 *Training History LSTM*  
👉 *(Insert screenshot here)*

---

### 5. CNN-LSTM Hybrid
- Combines CNN (local feature extraction) + LSTM (sequence memory).
- Often the strongest performer on sequential trading data.

📈 *Equity Curve CNN-LSTM*  
👉 *(Insert screenshot here)*

📉 *Training History CNN-LSTM*  
👉 *(Insert screenshot here)*

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
