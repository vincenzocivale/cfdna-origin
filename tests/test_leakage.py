"""Leakage checklist on a tiny synthetic dataset (docs/REFACTOR_PLAN.md: splits, fragments, representations, trainer).

Item (f) — test is revealed only after the best checkpoint is reloaded — is covered by test_smoke_end_to_end.py.
"""
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

import cfdna_origin.data.splits as splits_mod
from cfdna_origin.data.fragments import FragmentStore, finalize_dataset
from cfdna_origin.data.splits import LabelGuard, check_no_leakage, make_splits
from cfdna_origin.models.classifier import MILClassifier
from cfdna_origin.representations.base import NoLocusStore
from cfdna_origin.representations.controls import PositionOnlyStore, RandomLocusEmbeddingStore
from cfdna_origin.representations.hdf5 import HDF5LocusEmbeddingStore
from cfdna_origin.representations.registry import load_materialized, materialize
from cfdna_origin.training.trainer import LocusTable, train
from conftest import write_h5

SPLIT = dict(strategy="stratified_kfold", seed=3, n_folds=5, val_fraction=0.25, stratify_by=["label", "batch"])


def fragment_hashes(processed, sample_ids):
    """sha256 of each sample's fragment content (offsets, locus keys = loci[locus_row], states)."""
    loci = np.load(processed / "loci.npy")
    out = {}
    for s in sample_ids:
        d = processed / "fragments" / s
        h = hashlib.sha256()
        for arr in (np.load(d / "frag_offsets.npy"), loci[np.load(d / "locus_row.npy")], np.load(d / "state.npy")):
            h.update(arr.tobytes())
        out[s] = h.hexdigest()
    return out


def duplicated_fragments_across_splits(processed, fold_splits: pd.DataFrame) -> list[tuple[str, str]]:
    hashes = fragment_hashes(processed, fold_splits.sample_id.tolist())
    split_of = dict(zip(fold_splits.sample_id, fold_splits.split))
    by_hash: dict[str, list[str]] = {}
    for s, h in hashes.items():
        by_hash.setdefault(h, []).append(s)
    return [(a, b) for group in by_hash.values() for a in group for b in group
            if a < b and split_of[a] != split_of[b]]


def test_patients_and_samples_never_shared(tiny_dataset):
    _, samples = tiny_dataset
    splits = make_splits(samples, **SPLIT)
    check_no_leakage(splits)
    for _, g in splits.groupby("fold"):
        sets = {s: set(g[g.split == s].patient_id) for s in ("train", "val", "test")}
        assert not sets["train"] & sets["val"] and not sets["train"] & sets["test"] and not sets["val"] & sets["test"]
        assert not g.sample_id.duplicated().any() and set(g.sample_id) == set(samples.sample_id)


def test_no_fragment_content_shared_across_splits(tiny_dataset):
    root, samples = tiny_dataset
    splits = make_splits(samples, **SPLIT)
    for _, g in splits.groupby("fold"):
        assert duplicated_fragments_across_splits(root, g) == []


def test_duplicated_fragment_files_are_flagged(tiny_dataset):
    root, samples = tiny_dataset
    fold = make_splits(samples, **SPLIT).query("fold == 0")
    src = fold[fold.split == "train"].sample_id.iloc[0]
    dst = fold[fold.split == "test"].sample_id.iloc[0]
    for name in ("frag_offsets", "locus_row", "state"):  # e.g. the same run registered under two sample accessions
        shutil.copy(root / "fragments" / src / f"{name}.npy", root / "fragments" / dst / f"{name}.npy")
    assert duplicated_fragments_across_splits(root, fold) == [tuple(sorted((src, dst)))]


def test_representation_normalization_not_data_derived(tmp_path, tiny_dataset):
    root, _ = tiny_dataset
    loci = np.load(root / "loci.npy")
    rng = np.random.default_rng(0)
    h5 = write_h5(tmp_path / "f.h5", loci, rng.normal(size=(len(loci), 12)).astype(np.float32), chunks=(16, 12))
    stores = [NoLocusStore(), RandomLocusEmbeddingStore(dim=12), PositionOnlyStore(dim=32),
              HDF5LocusEmbeddingStore(h5, name="functional", normalization="store_sample_standardize"),
              HDF5LocusEmbeddingStore(h5, name="functional_raw", normalization="none")]
    for store in stores:
        m = materialize(store, loci, tmp_path / "cache" / store.name, dataset_manifest_sha="x",
                        projection={"type": "pca", "dim": 4, "fit_loci": 100, "seed": 0})
        assert m["normalization"]["data_derived"] is False, store.name
        assert m["projection"]["type"] == "none" if store.dim == 0 else m["projection"]["data_derived"] is False


def test_trainer_never_sees_test_samples(tmp_path, tiny_dataset, monkeypatch):
    root, samples = tiny_dataset
    fold = make_splits(samples, **SPLIT).query("fold == 0")
    guard = LabelGuard(samples, fold)
    tr, va = guard.samples("train"), guard.samples("val")
    test_ids = set(fold[fold.split == "test"].sample_id)
    store = FragmentStore(root, min_cpgs=3, max_cpgs=8)
    materialize(RandomLocusEmbeddingStore(dim=8), np.asarray(store.loci), tmp_path / "rep", dataset_manifest_sha="x")
    table = LocusTable(load_materialized(tmp_path / "rep", len(store.loci), "x")[0], "cpu")

    seen = set()
    orig_bag, orig_valid = FragmentStore.bag, FragmentStore.valid_fragments
    monkeypatch.setattr(FragmentStore, "bag", lambda self, ids, idx: (seen.update(ids), orig_bag(self, ids, idx))[1])
    monkeypatch.setattr(FragmentStore, "valid_fragments",
                        lambda self, sid, *a, **k: (seen.add(sid), orig_valid(self, sid, *a, **k))[1])
    torch.manual_seed(0)
    model = MILClassifier(8, 3, {"d_model": 16, "fragment_encoder": {"type": "deepsets", "hidden": 16},
                                 "aggregator": {"type": "gated_attention", "attn_dim": 8}})
    cfg = {"samples_per_step": 4, "fragments_per_bag": 8, "bags_per_sample_per_epoch": 1, "max_epochs": 2,
           "patience": 5, "lr": 1e-3, "weight_decay": 0.0, "val_max_fragments": 16, "eval_chunk": 8, "eval_seed": 0}
    res = train(model, store, table, train_ids=tr.sample_id.tolist(), train_labels=guard.labels("train"),
                val_ids=va.sample_id.tolist(), val_labels=guard.labels("val"), n_classes=3, cfg=cfg, seed=0,
                device="cpu", log=lambda *_: None)
    assert res["best_state"] is not None and len(res["history"]) == 2
    assert seen and seen <= set(tr.sample_id) | set(va.sample_id)
    assert not seen & test_ids
    with pytest.raises(splits_mod.TestLabelsHiddenError):  # test labels are unreachable before reveal
        guard.labels("test")


def test_finalize_is_label_free(tmp_path, tiny_dataset):
    """loci.npy is the union of fragment coordinates only: independent of sample order (and of labels)."""
    root, samples = tiny_dataset
    before = np.load(root / "loci.npy")
    finalize_dataset(root, samples.sample_id.tolist()[::-1], {"dataset": "tiny"})
    np.testing.assert_array_equal(np.load(root / "loci.npy"), before)
