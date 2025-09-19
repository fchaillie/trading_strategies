import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def compute_classification_metrics(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1], zero_division=0
    )
    return {
        "Precision_Down": precision[0],
        "Recall_Down": recall[0],
        "Precision_Up": precision[1],
        "Recall_Up": recall[1]
    }

def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()
