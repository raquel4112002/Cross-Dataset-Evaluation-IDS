from __future__ import annotations
import os, numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
from .utils import ensure_dir, save_json

def evaluate_binary(y_true, y_prob, y_pred, out_dir: str, prefix: str):
    ensure_dir(out_dir)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist()
    }
    save_json(metrics, os.path.join(out_dir, f"{prefix}_metrics.json"))

    # Confusion Matrix plot
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {prefix}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_cm.png"), bbox_inches="tight")
    plt.close(fig)
    return metrics
