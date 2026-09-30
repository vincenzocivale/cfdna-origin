import pandas as pd
import pytest

from cfdna_origin.config import load_component
from cfdna_origin.data.schema import REQUIRED_COLUMNS, SampleTableError, apply_label_scheme, validate_samples

DIAGNOSES = ["Normal", "CRC", "PDAC", "HCC", "GC", "ESCC", "EAC"]


def _table(diagnoses=DIAGNOSES):
    return pd.DataFrame({"sample_id": [f"S{i}" for i in range(len(diagnoses))],
                         "patient_id": [f"P{i}" for i in range(len(diagnoses))], "diagnosis": diagnoses,
                         "sample_type": "plasma_cfdna", "n_fragments": 10, "n_cpgs": 50})


def test_validate_samples_ok_and_filters_type():
    df = _table()
    df.loc[0, "sample_type"] = "tissue"
    out = validate_samples(df, sample_type="plasma_cfdna")
    assert len(out) == len(df) - 1 and out.index.tolist() == list(range(len(out)))


@pytest.mark.parametrize("col", REQUIRED_COLUMNS)
def test_validate_samples_missing_column(col):
    with pytest.raises(SampleTableError, match="lacks columns"):
        validate_samples(_table().drop(columns=[col]))


def test_validate_samples_duplicates_and_conflicts():
    df = _table()
    with pytest.raises(SampleTableError, match="duplicated"):
        validate_samples(pd.concat([df, df.iloc[:1]]))
    conflict = df.copy()
    conflict.loc[1, "patient_id"] = "P0"  # P0: Normal and CRC
    with pytest.raises(SampleTableError, match="conflicting"):
        validate_samples(conflict)
    missing = df.copy()
    missing.loc[2, "diagnosis"] = None
    with pytest.raises(SampleTableError, match="missing"):
        validate_samples(missing)


def test_gse149438_label_schemes():
    ds = load_component("datasets", "gse149438")
    merged, classes = apply_label_scheme(_table(), ds["label_schemes"]["esophageal_merged"])
    assert classes == ["Healthy", "Colorectal", "Pancreatic", "Hepatocellular", "Gastric", "Esophageal"]
    lab = dict(zip(merged.diagnosis, merged.label))
    assert lab["ESCC"] == lab["EAC"] == "Esophageal" and lab["Normal"] == "Healthy"
    assert (merged.label_idx == merged.label.map(classes.index)).all()
    split, classes = apply_label_scheme(_table(), ds["label_schemes"]["esophageal_split"])
    assert len(classes) == 7
    lab = dict(zip(split.diagnosis, split.label))
    assert lab["ESCC"] == "ESCC" and lab["EAC"] == "EAC"
    assert ds["label_scheme"] in ds["label_schemes"]


def test_label_scheme_errors():
    ds = load_component("datasets", "gse149438")
    with pytest.raises(SampleTableError, match="not covered"):
        apply_label_scheme(_table(DIAGNOSES + ["Melanoma"]), ds["label_schemes"]["esophageal_merged"])
    with pytest.raises(SampleTableError, match="not in `classes`"):
        apply_label_scheme(_table(["Normal"]), {"classes": ["A"], "map": {"Normal": "Healthy"}})
    out, _ = apply_label_scheme(_table(["Normal", "CRC"]), {"classes": ["Healthy"], "map": {"Normal": "Healthy",
                                                                                          "CRC": None}})
    assert out.diagnosis.tolist() == ["Normal"]  # explicit null -> excluded
