import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-weighted": f1_score(y_test, y_pred, average="weighted"),
        "F1-macro": f1_score(y_test, y_pred, average="macro"),
        "Precision-weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall-weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    if y_proba is not None:
        try:
            metrics["ROC-AUC-OVR"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )
        except ValueError:
            metrics["ROC-AUC-OVR"] = None
    else:
        metrics["ROC-AUC-OVR"] = None

    return metrics, y_pred, y_proba

def save_confusion_matrix(y_test, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_roc_curve(y_test, y_proba, labels, out_path: Path):
    if y_proba is None:
        return

    y_bin = label_binarize(y_test, classes=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, label in enumerate(labels):
        if y_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = _roc_curve_safe(y_bin[:, idx], y_proba[:, idx])
        ax.plot(fpr, tpr, label=f"{label}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _roc_curve_safe(y_true, y_score):
    from sklearn.metrics import roc_curve
    if np.unique(y_true).size < 2:
        return np.array([0, 1]), np.array([0, 1]), np.array([0.5])
    return roc_curve(y_true, y_score)

def save_metrics_summary(summary: dict, out_path: Path):
    out_path.write_text(json.dumps(summary, indent=2))
