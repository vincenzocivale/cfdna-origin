"""Patient-level, stratified, deterministic splits and the test-label guard.

A split table has one row per (fold, sample) with `split` in {train, val, test}. It depends only on the sample table
(patient ids and labels) and `split_seed` — never on a representation or training seed — and is written once with an
assignment hash; later runs must reuse it (`load_or_create_splits` refuses to silently change it).

Strategies:
- `stratified_kfold`: outer K folds over patients (each patient is test exactly once); validation is a stratified
  fraction of the remaining patients.
- `official`: use the dataset's `official_split` column (train/test); validation carved from train patients.
- `cross_batch`: fold k tests on batch k and trains on the other batch(es) (validation carved from the training
  batch): a batch-robustness protocol, only meaningful for classes present in every batch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class SplitError(ValueError):
    pass


def _patients(samples: pd.DataFrame, stratify_by: list[str]) -> pd.DataFrame:
    if samples.groupby("patient_id").label.nunique().max() > 1:
        raise SplitError("a patient has samples with different labels")
    p = samples.sort_values("sample_id").groupby("patient_id", sort=True).first().reset_index()
    p["stratum"] = p[stratify_by].astype(str).agg("|".join, axis=1)
    return p[["patient_id", "label", "stratum"]]


def _carve_val(train_patients: pd.DataFrame, val_fraction: float, seed: int) -> set:
    strat = None
    n_val = int(np.ceil(val_fraction * len(train_patients)))
    for col in ("stratum", "label"):  # finest stratification that train_test_split can honour
        vc = train_patients[col].value_counts()
        if vc.min() >= 2 and len(vc) <= min(n_val, len(train_patients) - n_val):
            strat = train_patients[col]; break
    _, val = train_test_split(train_patients.patient_id, test_size=val_fraction, random_state=seed, stratify=strat)
    return set(val)


def make_splits(samples: pd.DataFrame, *, strategy: str, seed: int, n_folds: int = 5,
                val_fraction: float = 0.2, stratify_by: list[str] | tuple = ("label",)) -> pd.DataFrame:
    pats = _patients(samples, list(stratify_by))
    rows = []
    if strategy == "stratified_kfold":
        counts = pats.label.value_counts()
        if counts.min() < n_folds:
            raise SplitError(f"class {counts.idxmin()!r} has {counts.min()} patients < n_folds={n_folds}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        vc = pats.stratum.value_counts()
        strata = pats.stratum.where(pats.stratum.map(vc) >= n_folds, pats.label)  # rare strata fall back to label
        for fold, (tr, te) in enumerate(skf.split(pats.patient_id, strata)):
            val = _carve_val(pats.iloc[tr], val_fraction, seed + fold)
            test = set(pats.patient_id.iloc[te])
            for pid in pats.patient_id:
                rows.append((fold, pid, "test" if pid in test else "val" if pid in val else "train"))
    elif strategy == "official":
        if "official_split" not in samples or samples.official_split.isna().any():
            raise SplitError("strategy 'official' needs a complete `official_split` column")
        off = samples.groupby("patient_id").official_split.agg(lambda s: set(s))
        if (off.map(len) > 1).any():
            raise SplitError("a patient appears in more than one official split")
        off = off.map(lambda s: next(iter(s)))
        if not set(off.unique()) <= {"train", "test"}:
            raise SplitError(f"official_split must be train/test, got {sorted(off.unique())}")
        val = _carve_val(pats[pats.patient_id.map(off) == "train"], val_fraction, seed)
        for pid in pats.patient_id:
            rows.append((0, pid, "test" if off[pid] == "test" else "val" if pid in val else "train"))
    elif strategy == "cross_batch":
        batch_of = samples.groupby("patient_id").batch.agg(lambda b: set(b))
        if (batch_of.map(len) > 1).any():
            raise SplitError("a patient has samples in several batches")
        batch_of = batch_of.map(lambda b: next(iter(b)))
        batches = sorted(batch_of.unique())
        per = samples.groupby(["label", "batch"]).patient_id.nunique().unstack(fill_value=0)
        if (per == 0).any().any():
            raise SplitError(f"cross_batch needs every class in every batch; missing combinations:\n{per}")
        for fold, test_batch in enumerate(batches):
            tr = pats[pats.patient_id.map(batch_of) != test_batch]
            val = _carve_val(tr.assign(stratum=tr.label), val_fraction, seed + fold)
            for pid in pats.patient_id:
                rows.append((fold, pid, "test" if batch_of[pid] == test_batch else "val" if pid in val else "train"))
    else:
        raise SplitError(f"unknown split strategy {strategy!r}")
    assign = pd.DataFrame(rows, columns=["fold", "patient_id", "split"])
    out = samples[["sample_id", "patient_id", "label"]].merge(assign, on="patient_id")
    return out.sort_values(["fold", "sample_id"]).reset_index(drop=True)


def confounded_classes(samples: pd.DataFrame, by: str = "batch") -> dict:
    """Classes observed in a single level of `by` (e.g. CRC only in KRp2): stratification cannot fix these."""
    per = samples.groupby("label")[by].agg(lambda b: sorted(set(b)))
    return {c: v[0] for c, v in per.items() if len(v) == 1}


def assignment_hash(splits: pd.DataFrame) -> str:
    s = splits.sort_values(["fold", "sample_id"])[["fold", "sample_id", "split"]].to_csv(index=False)
    return hashlib.sha256(s.encode()).hexdigest()


def check_no_leakage(splits: pd.DataFrame) -> None:
    for fold, g in splits.groupby("fold"):
        if g.sample_id.duplicated().any():
            raise SplitError(f"fold {fold}: duplicated samples")
        per_patient = g.groupby("patient_id").split.nunique()
        if (per_patient > 1).any():
            raise SplitError(f"fold {fold}: patients spanning splits: {per_patient[per_patient > 1].index[:5].tolist()}")
        if set(g.split) != {"train", "val", "test"}:
            raise SplitError(f"fold {fold}: missing split(s): {set(g.split)}")


def load_or_create_splits(path: Path, samples: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    splits = make_splits(samples, strategy=spec["strategy"], seed=spec["seed"], n_folds=spec.get("n_folds", 5),
                         val_fraction=spec.get("val_fraction", 0.2), stratify_by=spec.get("stratify_by", ["label"]))
    check_no_leakage(splits)
    manifest = {"spec": spec, "assignment_sha256": assignment_hash(splits), "n_samples": int(splits.sample_id.nunique()),
                "classes_confounded_with_batch": confounded_classes(samples) if "batch" in samples else {},
                "folds": {int(f): g.groupby("split").label.value_counts().unstack(fill_value=0).to_dict("index")
                          for f, g in splits.groupby("fold")}}
    if path.exists():
        old = json.loads((path.with_suffix(".json")).read_text())
        if old["assignment_sha256"] != manifest["assignment_sha256"]:
            raise SplitError(f"{path} exists with a different assignment (spec or sample table changed); "
                             "refusing to overwrite — delete it deliberately if the change is intended")
        return pd.read_parquet(path), old
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, default=int))
    return splits, manifest


class TestLabelsHiddenError(RuntimeError):
    pass


class LabelGuard:
    """Holds per-split labels; test labels are inaccessible until `reveal_test()` is called after model selection."""

    def __init__(self, samples: pd.DataFrame, fold_splits: pd.DataFrame):
        merged = samples.merge(fold_splits[["sample_id", "split"]], on="sample_id")
        self._by_split = {s: g.reset_index(drop=True) for s, g in merged.groupby("split")}
        self._revealed = False

    def samples(self, split: str) -> pd.DataFrame:
        """Sample table of a split. For test before reveal, label columns are removed."""
        df = self._by_split[split]
        if split == "test" and not self._revealed:
            return df.drop(columns=[c for c in ("label", "label_idx", "diagnosis") if c in df.columns])
        return df

    def labels(self, split: str) -> np.ndarray:
        if split == "test" and not self._revealed:
            raise TestLabelsHiddenError("test labels requested before model selection finished")
        return self._by_split[split].label_idx.to_numpy()

    def reveal_test(self) -> None:
        self._revealed = True
