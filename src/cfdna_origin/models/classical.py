"""Embedding-free baselines for the processed-beta pilot (fit on training samples only, inside each fold).

- `summary_only`: negative control on per-sample summary statistics (n observed CpGs, mean/variance of beta,
  coverage summary, missing fraction) -> standardised multinomial logistic regression.
- `dmr_logistic` / `dmr_xgboost`: EpiPanGI-style pipeline. Candidate regions = runs of consecutive panel CpGs with gaps
  <= `max_gap` bp (label-free, fixed for all folds); region beta = sum(methylated) / sum(coverage) over its observed
  CpGs. Inside each fold, on TRAINING samples only: keep regions observed in >= `min_observed_frac` of them, then per
  class one-vs-rest Welch t-test; keep the top `n_per_class` regions with |delta beta| >= `min_delta`; union ->
  fixed feature matrix -> classifier. Deviations from EpiPanGI (metilene DMR calling + Boruta + random forest) are
  documented in docs/BETA_BENCHMARK.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from cfdna_origin.data.loci import key_chrom_code, key_position


def candidate_regions(loci: np.ndarray, max_gap: int) -> np.ndarray:
    """Region id per locus: a new region starts at a chromosome change or a gap > max_gap bp."""
    chrom, pos = key_chrom_code(loci), key_position(loci)
    new = np.ones(len(loci), bool)
    new[1:] = (chrom[1:] != chrom[:-1]) | (pos[1:] - pos[:-1] > max_gap)
    return np.cumsum(new) - 1


def region_matrix(data, sample_ids: list[str], region_of: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(region beta [S, R] with NaN where unobserved, CpGs per region [R])."""
    n_reg = int(region_of.max()) + 1
    out = np.full((len(sample_ids), n_reg), np.nan, np.float32)
    for i, sid in enumerate(sample_ids):
        cols, b, c = data.tokens(sid, data.observed(sid))
        cov = np.bincount(region_of[cols], weights=c, minlength=n_reg)
        meth = np.bincount(region_of[cols], weights=b * c, minlength=n_reg)
        ok = cov > 0
        out[i, ok] = meth[ok] / cov[ok]
    return out, np.bincount(region_of, minlength=n_reg)


def select_dmrs(X: np.ndarray, y: np.ndarray, n_classes: int, *, n_per_class: int, min_delta: float,
                min_observed_frac: float) -> tuple[np.ndarray, dict]:
    observed = np.mean(~np.isnan(X), 0) >= min_observed_frac
    idx = np.flatnonzero(observed)
    Xo = X[:, idx]
    chosen, per_class = set(), {}
    for c in range(n_classes):
        a, b = Xo[y == c], Xo[y != c]
        if len(a) < 2:
            continue
        delta = np.nanmean(a, 0) - np.nanmean(b, 0)
        with np.errstate(all="ignore"):
            _, p = sps.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")
        p = np.asarray(np.ma.filled(p, np.nan), float)
        cand = np.flatnonzero((np.abs(delta) >= min_delta) & np.isfinite(p))
        top = cand[np.argsort(p[cand], kind="stable")[:n_per_class]]
        per_class[int(c)] = int(len(top))
        chosen.update(idx[top].tolist())
    sel = np.asarray(sorted(chosen), np.int64)
    return sel, {"n_candidate_regions": int(X.shape[1]), "n_regions_observed_enough": int(observed.sum()),
                 "n_selected_per_class": per_class, "n_dmrs": int(len(sel))}


class DMRClassifier:
    def __init__(self, kind: str, n_classes: int, cfg: dict, seed: int):
        self.kind, self.n_classes, self.cfg, self.seed = kind, n_classes, cfg, seed

    def fit(self, X_regions: np.ndarray, y: np.ndarray, region_sizes: np.ndarray) -> dict:
        self.sel, info = select_dmrs(X_regions, y, self.n_classes, n_per_class=self.cfg["n_per_class"],
                                     min_delta=self.cfg["min_delta"], min_observed_frac=self.cfg["min_observed_frac"])
        if not len(self.sel):
            raise ValueError("no DMR passed the selection thresholds")
        X = X_regions[:, self.sel]
        self.median = np.nanmedian(X, 0)
        w = compute_sample_weight("balanced", y)
        if self.kind == "dmr_logistic":
            params = dict(C=self.cfg.get("C", 1.0), max_iter=5000)
            self.model = make_pipeline(StandardScaler(), LogisticRegression(**params))
            self.model.fit(self._impute(X), y, logisticregression__sample_weight=w)
        elif self.kind == "dmr_xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError("dmr_xgboost needs xgboost (pip install xgboost)") from e
            params = dict(n_estimators=self.cfg.get("n_estimators", 300), max_depth=self.cfg.get("max_depth", 3),
                          learning_rate=self.cfg.get("learning_rate", 0.05), subsample=0.8, colsample_bytree=0.8,
                          objective="multi:softprob", num_class=self.n_classes, eval_metric="mlogloss", n_jobs=8,
                          random_state=self.seed, tree_method="hist")
            self.model = XGBClassifier(**params)
            self.model.fit(X, y, sample_weight=w)  # NaN handled natively (missing = unobserved)
        else:
            raise ValueError(self.kind)
        info.update(n_cpgs_in_dmrs=int(region_sizes[self.sel].sum()), classifier=self.kind, hyperparameters=params,
                    thresholds={k: self.cfg[k] for k in ("n_per_class", "min_delta", "min_observed_frac", "max_gap")},
                    imputation="training-fold median (logistic only)" if self.kind == "dmr_logistic" else "none (xgboost native)")
        return info

    def _impute(self, X):
        return np.where(np.isnan(X), self.median[None], X)

    def predict_proba(self, X_regions: np.ndarray) -> np.ndarray:
        X = X_regions[:, self.sel]
        return self.model.predict_proba(self._impute(X) if self.kind == "dmr_logistic" else X)


class SummaryClassifier:
    def __init__(self, seed: int):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    def fit(self, F: pd.DataFrame, y: np.ndarray) -> dict:
        self.model.fit(F.to_numpy(float), y, logisticregression__sample_weight=compute_sample_weight("balanced", y))
        return {"features": list(F.columns), "classifier": "standardised multinomial logistic regression, C=1"}

    def predict_proba(self, F: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(F.to_numpy(float))
