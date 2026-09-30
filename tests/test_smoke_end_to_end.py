"""Tiny CPU end-to-end smoke test: prepared dataset -> splits -> materialisation -> train -> test -> summary.

The fixture is synthetic (unit-test only): 3 classes x 6 patients, a few hundred fragments each; class-specific loci
are more methylated so that the pipeline has something to learn. It is not a benchmark.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cfdna_origin.config import load_experiment, load_paths
from cfdna_origin.data.fragments import finalize_dataset, write_sample
from cfdna_origin.data.loci import make_keys
from cfdna_origin.evaluation.summary import summarize
from cfdna_origin.experiments.runner import RUN_FILES, run_one

CLASSES = ["Healthy", "Colorectal", "Hepatocellular"]
DIAG = {"Healthy": "Normal", "Colorectal": "CRC", "Hepatocellular": "HCC"}


def _make_dataset(root, rng):
    loci = np.sort(np.unique(np.concatenate([make_keys("chr1", np.arange(1000, 1000 + 60 * 37, 37)),
                                             make_keys("chr2", np.arange(5000, 5000 + 60 * 41, 41))])))
    marker = {c: loci[i * 30 : (i + 1) * 30] for i, c in enumerate(CLASSES)}
    rows = []
    for c in CLASSES:
        for p in range(6):
            sid = f"S_{c[:3]}_{p}"
            offs, keys, states = [0], [], []
            for _ in range(150):
                start = rng.integers(0, len(loci) - 8)
                k = loci[start : start + rng.integers(3, 8)]
                meth = np.isin(k, marker[c])
                st = (rng.random(len(k)) < np.where(meth, 0.9, 0.2)).astype(np.uint8)
                keys.append(k); states.append(st); offs.append(offs[-1] + len(k))
            res = write_sample(root / "fragments" / sid, np.asarray(offs), np.concatenate(keys), np.concatenate(states))
            rows.append({"sample_id": sid, "patient_id": f"P_{sid}", "diagnosis": DIAG[c], "sample_type": "plasma_cfdna",
                         "batch": f"b{p % 2}", **res})
    samples = pd.DataFrame(rows)
    finalize_dataset(root, samples.sample_id.tolist(), {"dataset": "toy"})
    samples.to_parquet(root / "samples.parquet", index=False)
    return loci


def _make_store(path, loci, rng):
    # a "functional" store covering the dataset loci plus distractors, canonical contract
    extra = make_keys("chr3", np.arange(100, 100 + 500 * 13, 13))
    keys = np.sort(np.concatenate([loci, extra]))
    with h5py.File(path, "w") as f:
        f["cpg_idx"] = keys
        f.create_dataset("embedding", data=rng.normal(size=(len(keys), 16)).astype(np.float16), chunks=(64, 16))
        f.attrs["reference_build"] = "GRCh38"


@pytest.fixture()
def toy_experiment(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    loci = _make_dataset(data / "toy" / "processed", rng)
    _make_store(tmp_path / "functional.h5", loci, rng)
    cfg = tmp_path / "configs"
    for sub in ("datasets", "representations", "models", "experiments"):
        (cfg / sub).mkdir(parents=True)
    (cfg / "paths.yaml").write_text(yaml.safe_dump({
        "data_root": str(data), "reference_root": "{data_root}/reference", "cache_root": "{data_root}/cache",
        "outputs_root": str(tmp_path / "outputs"), "representations_root": str(tmp_path)}))
    (cfg / "datasets" / "toy.yaml").write_text(yaml.safe_dump({
        "name": "toy", "sample_type": "plasma_cfdna", "processed_dir": "{data_root}/toy/processed",
        "label_scheme": "s", "label_schemes": {"s": {"classes": CLASSES, "map": {v: k for k, v in DIAG.items()}}},
        "split": {"strategy": "stratified_kfold", "n_folds": 3, "val_fraction": 0.25, "seed": 1,
                  "stratify_by": ["label", "batch"]},
        "fragments": {"min_cpgs": 3, "max_cpgs": 6}}))
    reps = {"functional": {"kind": "hdf5", "path": "{representations_root}/functional.h5", "norm_sample_chunks": 4},
            "methylation_only": {"kind": "none"}, "random": {"kind": "random", "dim": 16}}
    for name, r in reps.items():
        (cfg / "representations" / f"{name}.yaml").write_text(yaml.safe_dump(r))
    (cfg / "models" / "tiny.yaml").write_text(yaml.safe_dump({
        "d_model": 16, "token": {"dropout": 0.1, "keep_min": 3},
        "fragment_encoder": {"type": "set_attention", "n_layers": 1, "n_heads": 2, "ffn_dim": 32, "readout": "pma"},
        "aggregator": {"type": "gated_attention", "attn_dim": 8, "class_branches": True, "instance_dropout": 0.1},
        "head_dropout": 0.1}))
    (cfg / "experiments" / "toy_main.yaml").write_text(yaml.safe_dump({
        "dataset": "toy", "model": "tiny", "representations": list(reps), "seeds": [3], "folds": [0, 1, 2],
        "representation_projection": {"type": "pca", "dim": 8, "fit_loci": 400, "seed": 0},
        "training": {"samples_per_step": 4, "fragments_per_bag": 64, "bags_per_sample_per_epoch": 1, "max_epochs": 3,
                     "patience": 2, "monitor": "val_macro_f1", "lr": 3e-3, "weight_decay": 0.01, "label_smoothing": 0.1,
                     "val_max_fragments": 100, "eval_chunk": 40, "eval_seed": 0},
        "unseen_locus": {"fraction": 0.2, "seed": 0},
        "comparisons": [["functional", "methylation_only"], ["functional", "random"]]}))
    exp = load_experiment("toy_main", config_dir=cfg)
    exp["_paths"] = load_paths(cfg)
    return exp


def test_end_to_end_cpu(toy_experiment):
    torch.set_num_threads(2)
    exp = toy_experiment
    dirs = [run_one(exp, rep, 3, fold, device="cpu", log=lambda *_: None)
            for rep in exp["representations"] for fold in exp["folds"]]
    for d in dirs:
        for f in RUN_FILES + ["RUN_COMPLETE.json"]:
            assert (d / f).exists(), f"{d}/{f}"
        m = json.loads((d / "metrics.json").read_text())
        assert np.isfinite(m["test/all"]["primary"]["macro_f1"])
        pred = pd.read_parquet(d / "predictions.parquet")
        need = {"sample_id", "true_label", "predicted_label", "number_of_reads", "number_of_valid_reads",
                "mean_CpGs_per_read", "representation", "seed", "split"} | {f"prob_{c}" for c in CLASSES}
        assert need <= set(pred.columns)
        assert "test/heldout_loci_only" in m  # unseen-locus protocol evaluated
        rep_manifest = json.loads((d / "representation_manifest.json").read_text())
        assert rep_manifest["normalization"].get("data_derived", False) is False
    # identical trainable parameters for arms sharing the projected width
    params = {d.parts[-3]: json.loads((d / "representation_manifest.json").read_text())["trainable_parameters"]
              for d in dirs}
    assert params["functional"] == params["random"]
    # every patient is predicted exactly once as test per (arm, seed) across folds
    exp_dir = dirs[0].parents[2]
    res = summarize(exp_dir, CLASSES, exp["seeds"], exp["folds"], [tuple(p) for p in exp["comparisons"]],
                    n_boot=50, n_perm=50)
    assert set(res["arms"].representation) == set(exp["representations"])
    assert not res["incomplete"]
    assert (res["per_seed"].n_patients == 18).all()
    assert set(res["per_batch"].batch) == {"b0", "b1"}
    assert {"delta", "ci95_low", "ci95_high", "p_perm_two_sided"} <= set(res["comparisons"].columns)


def test_rerun_skips_and_force_is_deterministic(toy_experiment):
    torch.set_num_threads(2)
    exp = toy_experiment
    d = run_one(exp, "functional", 3, 0, device="cpu", log=lambda *_: None)
    first = pd.read_parquet(d / "predictions.parquet")
    stamp = (d / "RUN_COMPLETE.json").stat().st_mtime_ns
    run_one(exp, "functional", 3, 0, device="cpu", log=lambda *_: None)  # skipped: complete
    assert (d / "RUN_COMPLETE.json").stat().st_mtime_ns == stamp
    run_one(exp, "functional", 3, 0, device="cpu", force=True, log=lambda *_: None)
    again = pd.read_parquet(d / "predictions.parquet")
    cols = [f"prob_{c}" for c in CLASSES]
    np.testing.assert_allclose(first[cols].to_numpy(), again[cols].to_numpy(), atol=1e-6)
