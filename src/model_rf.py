import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from .config import (N_ESTIMATORS, MAX_DEPTH, SEED, N_JOBS)

def fit_predict_block_rf(df: pd.DataFrame, train_idx, test_idx, feat_cols: List[str]):
    # Training rows with valid labels
    train_mask = df.loc[train_idx, "y"].notna()
    train_rows = df.loc[train_idx[train_mask]]

    X_train = train_rows[feat_cols].values
    y_train = train_rows["y"].values.astype(int)

    test_mask = df.loc[test_idx, "y"].notna()
    test_rows = df.loc[test_idx[test_mask]]
    X_test  = test_rows[feat_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Chronological sub-train / validation split
    n = len(X_train_s)
    split_idx = max(1, int(n * 0.8))
    X_subtrain, y_subtrain = X_train_s[:split_idx], y_train[:split_idx]
    X_valid, y_valid       = X_train_s[split_idx:], y_train[split_idx:]

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=SEED,
        n_jobs=N_JOBS,
        class_weight="balanced"
    )
    rf.fit(X_subtrain, y_subtrain)

    # Calibrate on the validation slice; rf already fitted => cv='prefit'
    calibrated_rf = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
    if len(X_valid) > 0:
        calibrated_rf.fit(X_valid, y_valid)

    # Probabilities on test set
    prob_up = calibrated_rf.predict_proba(X_test_s)[:, 1]

    p_up_test = pd.Series(index=test_idx, dtype=float, name="p_up")
    p_up_test.loc[test_rows.index] = prob_up

    # Return scaler too, in case downstream needs it
    return p_up_test, calibrated_rf, scaler

def plot_feature_importance(model: RandomForestClassifier, features: List[str]):
    import matplotlib.pyplot as plt
    import numpy as np
    if hasattr(model, "base_estimator"):
        est = model.base_estimator
    else:
        est = model
    if not hasattr(est, "feature_importances_"):
        raise ValueError("Provided model does not expose feature_importances_.")
    importances = est.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_feats = [features[i] for i in idx]
    sorted_imps = importances[idx]

    plt.figure(figsize=(8,6))
    plt.barh(sorted_feats[::-1], sorted_imps[::-1])
    plt.title("Feature Importances (RF)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
