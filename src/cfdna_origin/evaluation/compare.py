"""Paired comparison of representation arms at the PATIENT level.

Every arm predicts the same test patients, so differences are estimated by resampling patients (never reads):
- paired bootstrap: resample patients with replacement (stratified by true class so every class stays present),
  recompute each arm's metric on the same resample, report the mean difference and a percentile 95% CI;
- paired permutation test: under H0 (arms exchangeable) swap the two arms' predictions per patient at random.
With several training seeds, each arm's per-patient probabilities are averaged over seeds first (the seed ensemble
is the arm's prediction); per-seed metrics are reported separately in the summary. p-values are descriptive:
cohorts are small and several comparisons are made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfdna_origin.evaluation.metrics import PRIMARY, primary_metrics


def arm_probabilities(pred: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """predictions (all seeds/folds of one arm) -> (y [n], prob [n,C] averaged over seeds, sample ids)."""
    cols = [f"prob_{c}" for c in classes]
    g = pred.groupby("sample_id")
    counts = g.seed.nunique()
    if counts.nunique() > 1:
        raise ValueError("samples have predictions from different numbers of seeds; the arm is incomplete")
    prob = g[cols].mean()
    y = g.true_label.first().map({c: i for i, c in enumerate(classes)})
    return y.to_numpy(), prob.to_numpy(), prob.index.tolist()


def _stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in np.unique(y)])


def paired_bootstrap(y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, n_classes: int, *, n_boot: int = 2000,
                     seed: int = 0, metrics=PRIMARY) -> dict:
    rng = np.random.default_rng(seed)
    base_a, base_b = primary_metrics(y, prob_a, n_classes), primary_metrics(y, prob_b, n_classes)
    deltas = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = _stratified_resample(y, rng)
        ma, mb = primary_metrics(y[idx], prob_a[idx], n_classes), primary_metrics(y[idx], prob_b[idx], n_classes)
        for m in metrics:
            deltas[m].append(ma[m] - mb[m])
    out = {}
    for m in metrics:
        d = np.asarray(deltas[m], dtype=float)
        d = d[~np.isnan(d)]
        out[m] = {"a": base_a[m], "b": base_b[m], "delta": base_a[m] - base_b[m],
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [np.nan, np.nan],
                  "p_boot_delta_le_0": float((d <= 0).mean()) if len(d) else np.nan}
    return out


def paired_permutation(y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, n_classes: int, *, n_perm: int = 2000,
                       seed: int = 0, metrics=PRIMARY) -> dict:
    rng = np.random.default_rng(seed)
    obs = {m: primary_metrics(y, prob_a, n_classes)[m] - primary_metrics(y, prob_b, n_classes)[m] for m in metrics}
    ge = {m: 0 for m in metrics}
    for _ in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        pa = np.where(swap[:, None], prob_b, prob_a); pb = np.where(swap[:, None], prob_a, prob_b)
        ma, mb = primary_metrics(y, pa, n_classes), primary_metrics(y, pb, n_classes)
        for m in metrics:
            ge[m] += abs(ma[m] - mb[m]) >= abs(obs[m]) - 1e-12
    return {m: {"delta": obs[m], "p_two_sided": (ge[m] + 1) / (n_perm + 1)} for m in metrics}


def compare_arms(predictions: pd.DataFrame, classes: list[str], pairs: list[tuple[str, str]], *, n_boot: int = 2000,
                 n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    if "patient_id" in predictions and (predictions.groupby("patient_id").sample_id.nunique() > 1).any():
        raise NotImplementedError("several samples per patient: aggregate to patients before comparing arms")
    rows = []
    arms = {}
    for rep, g in predictions.groupby("representation"):
        arms[rep] = arm_probabilities(g, classes)
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        ya, pa, ida = arms[a]; yb, pb, idb = arms[b]
        if ida != idb or not np.array_equal(ya, yb):
            raise ValueError(f"{a} and {b} were not evaluated on the same patients")
        boot = paired_bootstrap(ya, pa, pb, len(classes), n_boot=n_boot, seed=seed)
        perm = paired_permutation(ya, pa, pb, len(classes), n_perm=n_perm, seed=seed)
        for m in boot:
            rows.append({"arm_a": a, "arm_b": b, "metric": m, "a": boot[m]["a"], "b": boot[m]["b"],
                         "delta": boot[m]["delta"], "ci95_low": boot[m]["ci95"][0], "ci95_high": boot[m]["ci95"][1],
                         "p_perm_two_sided": perm[m]["p_two_sided"], "n_patients": len(ya)})
    return pd.DataFrame(rows)
