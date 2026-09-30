"""Sample-table schema shared by all dataset adapters, and label mapping.

Every processed dataset has `samples.parquet` with at least `REQUIRED_COLUMNS`. `patient_id` is the unit of splitting
(several samples of one patient never span splits). `diagnosis` is the raw clinical label as published; `label` is
derived from it through the dataset's configured label scheme, never from anything computed on the data.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["sample_id", "patient_id", "diagnosis", "sample_type", "n_fragments", "n_cpgs"]
OPTIONAL_COLUMNS = ["batch", "official_split", "sex", "age", "stage", "source_accession", "notes"]


class SampleTableError(ValueError):
    pass


def validate_samples(df: pd.DataFrame, *, sample_type: str | None = None) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SampleTableError(f"samples table lacks columns {missing}")
    if df.sample_id.duplicated().any():
        raise SampleTableError(f"duplicated sample_id: {df.sample_id[df.sample_id.duplicated()].tolist()[:5]}")
    if df[["sample_id", "patient_id", "diagnosis"]].isna().any().any():
        raise SampleTableError("sample_id / patient_id / diagnosis must not be missing")
    per_patient = df.groupby("patient_id").diagnosis.nunique()
    if (per_patient > 1).any():
        raise SampleTableError(f"patients with conflicting diagnoses: {per_patient[per_patient > 1].index.tolist()[:5]}")
    if sample_type is not None:
        df = df[df.sample_type == sample_type]
    return df.reset_index(drop=True)


def apply_label_scheme(df: pd.DataFrame, scheme: dict) -> tuple[pd.DataFrame, list[str]]:
    """scheme = {"classes": [...], "map": {diagnosis: class | null}}; null/absent diagnoses are excluded explicitly."""
    classes = list(scheme["classes"])
    mapping = scheme["map"]
    unknown = sorted(set(df.diagnosis) - set(mapping))
    if unknown:
        raise SampleTableError(f"diagnoses not covered by the label scheme: {unknown}")
    out = df.assign(label=df.diagnosis.map(mapping))
    out = out[out.label.notna()].reset_index(drop=True)
    bad = sorted(set(out.label) - set(classes))
    if bad:
        raise SampleTableError(f"label scheme maps to classes not in `classes`: {bad}")
    out["label_idx"] = out.label.map({c: i for i, c in enumerate(classes)}).astype(int)
    return out, classes
