# Random Forest Strategy — Walk-Forward + Trade Log + Feature Importance

This repo packages your RF-based strategy with:
- Walk-forward style splits (anchored window utility)
- Probability masking (by day and series ends)
- RF + isotonic calibration (validation slice)
- 5‑minute decision, tick-level execution backtest (SL + Trailing)
- Trade log CSVs and equity plots

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run
python scripts/run_main.py   --parquet MBT_bars_5m_quant_PARIS_tz.parquet   --ticks_parquet ticks.parquet  # optional; if omitted, synthetic ticks are generated
```
Outputs: CSV trade logs, equity curve plots under `outputs/`.

> Note: I removed the non-standard `sklearn.frozen.FrozenEstimator` import and used calibrated
> probabilities with `CalibratedClassifierCV(cv="prefit")` after fitting RF on a subtrain split.
