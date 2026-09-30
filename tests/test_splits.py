import pandas as pd
import pytest

import cfdna_origin.data.splits as splits_mod
from cfdna_origin.data.splits import (
    LabelGuard,
    SplitError,
    check_no_leakage,
    load_or_create_splits,
    make_splits,
)

KFOLD = dict(strategy="stratified_kfold", seed=7, n_folds=5, val_fraction=0.2, stratify_by=["label", "batch"])


def test_stratified_kfold_properties(tiny_samples):
    sp = make_splits(tiny_samples, **KFOLD)
    check_no_leakage(sp)
    test_counts = sp[sp.split == "test"].patient_id.value_counts()
    assert set(test_counts.index) == set(tiny_samples.patient_id) and (test_counts == 1).all()
    for _, g in sp.groupby("fold"):
        assert len(g) == len(tiny_samples) and (g.groupby("patient_id").split.nunique() == 1).all()
        assert (g.groupby(["split", "label"]).size().unstack().loc["test"] == 2).all()  # 10 patients/class, 5 folds
        n_nontest = (g.split != "test").sum()
        assert abs((g.split == "val").sum() - 0.2 * n_nontest) <= 3
        for split in ("train", "test"):  # batch balance is kept roughly within each split
            b = g[g.split == split].merge(tiny_samples[["sample_id", "batch"]]).batch.value_counts()
            assert b.max() - b.min() <= 3


def test_kfold_deterministic_and_seed_dependent(tiny_samples):
    a, b = make_splits(tiny_samples, **KFOLD), make_splits(tiny_samples, **KFOLD)
    pd.testing.assert_frame_equal(a, b)
    c = make_splits(tiny_samples, **{**KFOLD, "seed": 8})
    assert not a.split.equals(c.split)


def test_kfold_too_few_patients(tiny_samples):
    with pytest.raises(SplitError):
        make_splits(tiny_samples, **{**KFOLD, "n_folds": 11})


def test_several_samples_per_patient_stay_together(tiny_samples):
    extra = tiny_samples.iloc[:6].assign(sample_id=lambda d: d.sample_id + "_b")  # second sample of 6 patients
    sp = make_splits(pd.concat([tiny_samples, extra], ignore_index=True), **KFOLD)
    check_no_leakage(sp)
    assert len(sp) == 5 * 36


def test_official_strategy(tiny_samples):
    s = tiny_samples.assign(official_split=["test" if i % 3 == 0 else "train" for i in range(len(tiny_samples))])
    sp = make_splits(s, strategy="official", seed=1, val_fraction=0.25, stratify_by=["label"])
    check_no_leakage(sp)
    assert set(sp.fold) == {0}
    m = sp.merge(s[["sample_id", "official_split"]])
    assert ((m.split == "test") == (m.official_split == "test")).all()
    assert (m[m.split == "val"].official_split == "train").all()
    with pytest.raises(SplitError):
        make_splits(s.assign(official_split=None), strategy="official", seed=1)
    with pytest.raises(SplitError):
        make_splits(s.assign(official_split="validation"), strategy="official", seed=1)


def test_load_or_create_splits_refuses_overwrite(tmp_path, tiny_samples):
    spec = dict(KFOLD)
    path = tmp_path / "splits" / "kfold.parquet"
    sp, man = load_or_create_splits(path, tiny_samples, spec)
    assert path.exists() and path.with_suffix(".json").exists()
    sp2, man2 = load_or_create_splits(path, tiny_samples, spec)
    pd.testing.assert_frame_equal(sp.reset_index(drop=True), sp2)
    assert man2["assignment_sha256"] == man["assignment_sha256"]
    with pytest.raises(SplitError, match="refusing to overwrite"):
        load_or_create_splits(path, tiny_samples, {**spec, "seed": 99})


def test_check_no_leakage_catches_problems(tiny_samples):
    sp = make_splits(tiny_samples, **KFOLD)
    f0 = sp[sp.fold == 0]
    check_no_leakage(f0)
    pid = f0[f0.split == "test"].patient_id.iloc[0]
    other = f0[f0.patient_id == pid].assign(sample_id="S_new", split="train")  # same patient, another split
    with pytest.raises(SplitError, match="spanning"):
        check_no_leakage(pd.concat([f0, other]))
    with pytest.raises(SplitError, match="duplicated"):
        check_no_leakage(pd.concat([f0, f0.iloc[:1]]))
    with pytest.raises(SplitError, match="missing"):
        check_no_leakage(f0[f0.split != "val"])


def test_label_guard(tiny_samples):
    sp = make_splits(tiny_samples, **KFOLD)
    guard = LabelGuard(tiny_samples, sp[sp.fold == 0][["sample_id", "split"]])
    assert len(guard.labels("train")) == len(guard.samples("train")) and len(guard.labels("val"))
    with pytest.raises(splits_mod.TestLabelsHiddenError):
        guard.labels("test")
    hidden = guard.samples("test")
    assert not {"label", "label_idx", "diagnosis"} & set(hidden.columns) and len(hidden) == 6
    guard.reveal_test()
    assert {"label", "label_idx", "diagnosis"} <= set(guard.samples("test").columns)
    assert len(guard.labels("test")) == 6
