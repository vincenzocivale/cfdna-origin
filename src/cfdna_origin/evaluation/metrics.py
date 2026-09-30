"""Sample-level classification metrics (never read-level)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

PRIMARY = ("macro_f1", "balanced_accuracy", "auroc_macro", "accuracy")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _auroc_ovr(y: np.ndarray, prob: np.ndarray, c: int) -> float:
    pos = y == c
    if pos.all() or not pos.any():
        return float("nan")
    return float(roc_auc_score(pos, prob[:, c]))


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    conf = prob.max(1); correct = prob.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def primary_metrics(y: np.ndarray, prob: np.ndarray, n_classes: int) -> dict:
    """Cheap subset used inside bootstrap loops."""
    pred = prob.argmax(1)
    labels = np.arange(n_classes)
    aucs = [_auroc_ovr(y, prob, c) for c in labels]
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc_macro": float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan"),
        "accuracy": float((pred == y).mean()),
    }


def all_metrics(y: np.ndarray, prob: np.ndarray, classes: list[str]) -> dict:
    n = len(classes)
    y = np.asarray(y); prob = np.asarray(prob, dtype=np.float64)
    if prob.shape != (len(y), n):
        raise ValueError(f"probabilities {prob.shape} vs {len(y)} samples x {n} classes")
    if not np.allclose(prob.sum(1), 1.0, atol=1e-4):
        raise ValueError("probabilities do not sum to 1")
    pred = prob.argmax(1)
    labels = np.arange(n)
    cm = confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = cm.sum() - tp - fn - fp
    top2 = np.argsort(-prob, 1)[:, :2]
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = tp / (tp + fn); spec = tn / (tn + fp)
    per_class = {c: {"n": int(cm[i].sum()), "sensitivity": float(sens[i]), "specificity": float(spec[i]),
                     "auroc": _auroc_ovr(y, prob, i),
                     "f1": float(f1_score(y == i, pred == i, zero_division=0))} for i, c in enumerate(classes)}
    return {
        "n_samples": int(len(y)),
        "primary": primary_metrics(y, prob, n),
        "secondary": {
            "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
            "top2_accuracy": float((top2 == y[:, None]).any(1).mean()),
            "ece": expected_calibration_error(y, prob),
            "log_loss": float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None)))),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
    }
