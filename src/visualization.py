import matplotlib.pyplot as plt
import numpy as np

def plot_equity_curve(equity, model_name, H, P_LONG, P_SHORT, save_path=None):
    plt.figure(figsize=(6,3))
    plt.plot(equity.index, equity.values, label="Equity", color="blue")
    plt.title(f"Equity Curve {model_name} H={H} LONG={P_LONG} SHORT={P_SHORT}")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, features, save_path=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        sorted_feats = [features[i] for i in idx]
        sorted_imps = importances[idx]
        plt.figure(figsize=(8,6))
        plt.barh(sorted_feats[::-1], sorted_imps[::-1])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

def plot_training_history(history, model_name, save_path=None):
    if history is None:
        return
    plt.figure(figsize=(6,3))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"Training History {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
