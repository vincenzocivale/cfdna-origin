"""Processed-beta runner: task/protocol sample tables, experiment configs, the functional_shuffled control and a tiny
CPU end-to-end of `run_beta` (synthetic data only; every file under tmp_path, a few KB)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import cfdna_origin.experiments.beta_runner as br
from cfdna_origin.config import CONFIG_DIR, load_component, load_experiment, load_paths, read_yaml
from cfdna_origin.data.loci import make_keys
from cfdna_origin.data.splits import LabelGuard
from cfdna_origin.evaluation.summary import summarize
from cfdna_origin.experiments.runner import RUN_FILES
from cfdna_origin.representations.hdf5 import HDF5LocusEmbeddingStore
from cfdna_origin.representations.registry import (
    RepresentationUnavailableError,
    load_materialized,
    materialize,
)
from test_beta_data import write_processed

# ---------------------------------------------------------------- task_samples (protocols / tasks / label schemes)
REAL_DS = load_component("datasets", "gse149438_processed_beta")


def _samples(counts):
    """counts: {(diagnosis, batch): n_patients} -> sample table (one sample per patient)."""
    rows = []
    for (diag, batch), n in counts.items():
        for p in range(n):
            sid = f"{diag}_{batch}_{p}"
            rows.append({"sample_id": sid, "patient_id": f"P_{sid}", "diagnosis": diag, "batch": batch})
    return SimpleNamespace(samples=pd.DataFrame(rows))


COUNTS = {("Normal", "KRp1"): 3, ("Normal", "KRp2"): 3, ("CRC", "KRp2"): 4, ("GC", "KRp1"): 2, ("GC", "KRp2"): 1,
          ("HCC", "KRp1"): 2, ("HCC", "KRp2"): 2, ("ESCC", "KRp1"): 1, ("EAC", "KRp2"): 1, ("PDAC", "KRp1"): 1}


def _exp(protocol, task="tissue", scheme="esophageal_merged"):
    return {"dataset": {**REAL_DS, "label_scheme": scheme}, "protocol": protocol, "task": task}


def test_task_samples_all_classes_merged_vs_split():
    data = _samples(COUNTS)
    df, classes = br.task_samples(_exp({"name": "all_classes_original"}), data)
    assert classes == ["Healthy", "Colorectal", "Pancreatic", "Hepatocellular", "Gastric", "Esophageal"]
    assert (df.label == "Esophageal").sum() == 2 and len(df) == 20
    assert (df.label_idx == df.label.map({c: i for i, c in enumerate(classes)})).all()
    df2, classes2 = br.task_samples(_exp({"name": "all_classes_original"}, scheme="esophageal_split"), data)
    assert classes2[-2:] == ["ESCC", "EAC"] and len(classes2) == 7
    assert set(df2.label) == set(classes2)


def test_task_samples_exclude_crc():
    df, classes = br.task_samples(_exp({"name": "exclude_crc", "exclude_classes": ["Colorectal"]}), _samples(COUNTS))
    assert "Colorectal" not in classes and "Colorectal" not in set(df.label) and len(df) == 16
    assert sorted(df.label_idx.unique()) == list(range(len(classes)))


def test_task_samples_batch_robust_subset():
    df, classes = br.task_samples(_exp({"name": "batch_robust_subset", "min_per_batch": 2}), _samples(COUNTS))
    assert classes == ["Healthy", "Hepatocellular"]  # GC has 1 patient in KRp2, CRC none in KRp1
    per = df.groupby(["label", "batch"]).patient_id.nunique()
    assert (per >= 2).all() and len(per) == 4
    df3, classes3 = br.task_samples(_exp({"name": "batch_robust_subset", "min_per_batch": 3}), _samples(COUNTS))
    assert classes3 == ["Healthy"]


def test_task_samples_batch_task():
    df, classes = br.task_samples(_exp({"name": "exclude_crc", "exclude_classes": ["Colorectal"]}, task="batch"),
                                  _samples(COUNTS))
    assert classes == ["KRp1", "KRp2"]
    assert (df.label == df.batch).all() and set(df.label_idx) == {0, 1}
    assert "tissue_label" in df and "Colorectal" not in set(df.tissue_label)
    assert set(df.tissue_label) == {"Healthy", "Gastric", "Hepatocellular", "Esophageal", "Pancreatic"}


# ---------------------------------------------------------------- the committed beta experiment configs
def test_beta_experiment_configs():
    common = read_yaml(CONFIG_DIR / "experiments" / "_beta_common.yaml")
    reps = {p.stem for p in (CONFIG_DIR / "representations").glob("*.yaml")}
    for path in sorted((CONFIG_DIR / "experiments").glob("beta_*.yaml")):
        raw = read_yaml(path)
        assert raw["base"] == "_beta_common", path
        assert set(raw["representations"]) <= reps, path
        assert set(raw.get("baseline_arms", [])) <= set(br.BASELINES), path
        for a, b in raw.get("comparisons", common["comparisons"]):
            assert {a, b} <= reps | set(br.BASELINES), path
    exp = load_experiment("beta_archsel_pma")
    assert "base" not in exp and exp["selection_only"] is True
    assert exp["seeds"] == [17] and exp["folds"] == common["folds"]  # experiment overrides base; base fills the rest
    assert exp["training"] == common["training"] and exp["model"]["arch"] == "pma"
    assert exp["dataset"]["mode"] == "processed_beta" and exp["dataset"]["split"]["seed_from_run"] is True
    robust = read_yaml(CONFIG_DIR / "experiments" / "beta_batch_robust.yaml")
    assert robust["protocol"]["split"]["strategy"] == "cross_batch" and robust["folds"] == [0, 1]


# ---------------------------------------------------------------- functional_shuffled control
def test_shuffled_materialisation_is_exact_permutation(tmp_path):
    rng = np.random.default_rng(0)
    universe = np.sort(np.concatenate([make_keys("chr1", np.arange(1, 301) * 10), make_keys("chr2", np.arange(1, 201) * 7)]))
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f["cpg_idx"] = universe
        f.create_dataset("embedding", data=rng.normal(size=(len(universe), 12)).astype(np.float32), chunks=(50, 12))
    store = lambda: HDF5LocusEmbeddingStore(tmp_path / "f.h5", name="functional", normalization="none")  # noqa: E731
    loci = universe[::3]
    proj = {"type": "pca", "dim": 6, "fit_loci": 200, "seed": 0}
    kw = dict(dataset_manifest_sha="abc", projection=proj)
    m0 = materialize(store(), loci, tmp_path / "plain", **kw)
    m1 = materialize(store(), loci, tmp_path / "s1", shuffle_loci_seed=1, **kw)
    m1b = materialize(store(), loci, tmp_path / "s1b", shuffle_loci_seed=1, **kw)
    m2 = materialize(store(), loci, tmp_path / "s2", shuffle_loci_seed=2, **kw)
    t0 = load_materialized(tmp_path / "plain", len(loci), "abc")[0]
    t1 = load_materialized(tmp_path / "s1", len(loci), "abc")[0]
    t1b = load_materialized(tmp_path / "s1b", len(loci), "abc")[0]
    t2 = load_materialized(tmp_path / "s2", len(loci), "abc")[0]
    # rows are unique, so an exact multiset match of rows means a bijection (no row duplicated or lost)
    assert len(np.unique(t0, axis=0)) == len(loci)
    np.testing.assert_array_equal(np.unique(t1, axis=0), np.unique(t0, axis=0))
    assert len(np.unique(t1, axis=0)) == len(loci)
    row_of = {r.tobytes(): i for i, r in enumerate(np.asarray(t0))}
    perm = np.array([row_of[r.tobytes()] for r in np.asarray(t1)])
    assert sorted(perm) == list(range(len(loci))) and (perm != np.arange(len(loci))).mean() > 0.9
    np.testing.assert_array_equal(perm, np.random.default_rng(1).permutation(len(loci)))
    np.testing.assert_array_equal(t1, t1b)  # deterministic for the seed
    assert not np.array_equal(t1, t2)
    # the projection is fitted on the store, never on the dataset rows: identical for plain and shuffled arms
    assert m0["projection"] == m1["projection"] == m2["projection"]
    assert m0["projection"]["components_sha256"] == m1["projection"]["components_sha256"]
    assert m1["shuffle_loci_seed"] == 1 and m0["shuffle_loci_seed"] is None
    assert m1["materialization"]["table_sha256"] == m1b["materialization"]["table_sha256"]
    assert m1["materialization"]["table_sha256"] != m0["materialization"]["table_sha256"]
    # loading twice (e.g. training then test time) yields the identical table
    np.testing.assert_array_equal(t1, load_materialized(tmp_path / "s1", len(loci), "abc")[0])


# ---------------------------------------------------------------- tiny CPU end-to-end of run_beta
CLASSES = ["Healthy", "Colorectal", "Gastric"]
DIAG = {"Healthy": "Normal", "Colorectal": "CRC", "Gastric": "GC"}
ARMS = ["methylation_only", "random", "functional", "functional_shuffled"]
BASE_ARMS = ["summary_only", "dmr_logistic"]
SEED, FOLDS = 3, [0, 1, 2]
QUIET = dict(device="cpu", log=lambda *_: None)


def _make_beta_dataset(root, rng):
    # 30 clusters x 10 CpGs (20 bp apart, clusters 1 kb apart) -> 30 DMR candidate regions at max_gap 100
    pos = (np.arange(30)[:, None] * 1000 + np.arange(10)[None] * 20 + 100).ravel()
    loci = np.concatenate([make_keys("chr1", pos[:150]), make_keys("chr2", pos[150:])])
    cluster = np.arange(len(loci)) // 10
    rows, betas, covs = [], [], []
    for k, c in enumerate(CLASSES):
        for p in range(6):
            cov = rng.integers(4, 40, len(loci)) * (rng.random(len(loci)) > 0.3)
            beta = np.where(cluster // 4 == k, 0.85, 0.15) + rng.normal(0, 0.05, len(loci))
            betas.append(np.where(cov > 0, np.clip(beta, 0, 1), np.nan)); covs.append(cov)
            rows.append({"sample_id": f"GSM{k}{p}", "patient_id": f"P{k}{p}", "diagnosis": DIAG[c],
                         "batch": f"KRp{p % 2 + 1}", "sample_type": "plasma_cfdna"})
    write_processed(root, loci, np.array(betas), np.array(covs), pd.DataFrame(rows),
                    {"dataset": "tiny_beta", "mode": "processed_beta", "n_loci": int(len(loci)),
                     "parse_stats": {"GSM00": {"rows": 1}}})
    return loci


def _write_configs(tmp, loci, rng):
    extra = make_keys("chr3", np.arange(1, 401) * 13)
    keys = np.sort(np.concatenate([loci, extra]))
    with h5py.File(tmp / "functional.h5", "w") as f:
        f["cpg_idx"] = keys
        f.create_dataset("embedding", data=rng.normal(size=(len(keys), 16)).astype(np.float16), chunks=(64, 16))
    cfg = tmp / "configs"
    for sub in ("datasets", "representations", "models", "experiments"):
        (cfg / sub).mkdir(parents=True)
    (cfg / "paths.yaml").write_text(yaml.safe_dump({
        "data_root": str(tmp / "data"), "reference_root": "{data_root}/reference", "cache_root": str(tmp / "cache"),
        "outputs_root": str(tmp / "outputs"), "representations_root": str(tmp)}))
    (cfg / "datasets" / "tiny_beta.yaml").write_text(yaml.safe_dump({
        "name": "tiny_beta", "mode": "processed_beta", "sample_type": "plasma_cfdna",
        "processed_dir": "{data_root}/tiny_beta", "label_scheme": "s",
        "label_schemes": {"s": {"classes": CLASSES, "map": {v: k for k, v in DIAG.items()}}},
        "split": {"strategy": "stratified_kfold", "n_folds": 3, "val_fraction": 0.25, "seed_from_run": True,
                  "stratify_by": ["label", "batch"]}}))
    h5 = {"kind": "hdf5", "path": "{representations_root}/functional.h5", "id_key": "cpg_idx",
          "embedding_key": "embedding", "normalization": "none"}
    reps = {"methylation_only": {"kind": "none"}, "random": {"kind": "random", "dim": 16}, "functional": h5,
            "functional_shuffled": {**h5, "shuffle_loci_seed": 1},
            "missing_store": {"kind": "hdf5", "path": "{representations_root}/nope.h5", "availability": "not built"}}
    for name, r in reps.items():
        (cfg / "representations" / f"{name}.yaml").write_text(yaml.safe_dump(r))
    (cfg / "models" / "tiny_pma.yaml").write_text(yaml.safe_dump(
        {"arch": "pma", "d_model": 8, "hidden": 16, "n_heads": 4, "dropout": 0.1, "head_dropout": 0.1,
         "use_coverage": True}))
    (cfg / "experiments" / "_tiny_common.yaml").write_text(yaml.safe_dump({
        "dataset": "tiny_beta", "representation_projection": {"type": "pca", "dim": 8, "fit_loci": 300, "seed": 0},
        "seeds": [SEED], "folds": FOLDS,
        "training": {"samples_per_step": 4, "tokens_per_sample": 32, "bags_per_sample_per_epoch": 1, "max_epochs": 2,
                     "patience": 2, "monitor": "val_macro_f1", "lr": 3e-3, "weight_decay": 0.01, "label_smoothing": 0.1,
                     "val_max_tokens": 64, "eval_chunk": 50, "eval_seed": 0},
        "baselines": {"summary_only": {},
                      "dmr_logistic": {"max_gap": 100, "min_observed_frac": 0.5, "n_per_class": 3, "min_delta": 0.1,
                                       "C": 1.0}},
        "comparisons": [["functional", a] for a in ARMS[:2] + ARMS[3:] + BASE_ARMS]}))
    (cfg / "experiments" / "tiny_main.yaml").write_text(yaml.safe_dump({
        "base": "_tiny_common", "model": "tiny_pma", "protocol": {"name": "all_classes_original"},
        "representations": ARMS + ["missing_store"], "baseline_arms": BASE_ARMS,
        "attribution": {"arms": ["functional"], "seed": SEED, "attribution_top_k": 5}}))
    (cfg / "experiments" / "tiny_archsel.yaml").write_text(yaml.safe_dump({
        "base": "_tiny_common", "model": "tiny_pma", "protocol": {"name": "all_classes_original"},
        "representations": ["methylation_only"], "baseline_arms": [], "folds": [0], "selection_only": True}))
    return cfg


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    torch.set_num_threads(2)
    tmp = tmp_path_factory.mktemp("beta_e2e")
    rng = np.random.default_rng(0)
    loci = _make_beta_dataset(tmp / "data" / "tiny_beta", rng)
    cfg = _write_configs(tmp, loci, rng)
    paths = load_paths(cfg)
    exp = load_experiment("tiny_main", config_dir=cfg); exp["_paths"] = paths
    sel = load_experiment("tiny_archsel", config_dir=cfg); sel["_paths"] = paths
    fit_calls = []
    orig_region_matrix = br.region_matrix
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(br, "region_matrix", lambda d, ids, reg: fit_calls.append(list(ids)) or orig_region_matrix(d, ids, reg))
        dirs = {(arm, f): br.run_beta(exp, arm, SEED, f, **QUIET) for arm in ARMS + BASE_ARMS for f in FOLDS}

    def no_reveal(self):
        raise AssertionError("reveal_test called in a selection_only experiment")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LabelGuard, "reveal_test", no_reveal)
        sel_dir = br.run_beta(sel, "methylation_only", SEED, 0, **QUIET)
    yield SimpleNamespace(exp=exp, dirs=dirs, sel_dir=sel_dir, fit_calls=fit_calls, loci=loci, cfg=cfg)
    br._DATA_CACHE.clear()


def test_e2e_run_files_and_prediction_columns(e2e):
    for (arm, fold), d in e2e.dirs.items():
        for f in RUN_FILES + ["RUN_COMPLETE.json"]:
            assert (d / f).exists(), f"{d}/{f}"
        pred = pd.read_parquet(d / "predictions.parquet")
        assert {"batch", "n_observed_loci", "missing_fraction", "mean_coverage", "sample_id", "patient_id",
                "true_label", "predicted_label"} | {f"prob_{c}" for c in CLASSES} <= set(pred.columns)
        assert set(pred.split) == {"val", "test"}
        np.testing.assert_allclose(pred[[f"prob_{c}" for c in CLASSES]].sum(1), 1.0, atol=1e-5)
        m = json.loads((d / "metrics.json").read_text())
        assert np.isfinite(m["test/all"]["primary"]["macro_f1"]) and m["run"]["representation"] == arm
        dm = json.loads((d / "dataset_manifest.json").read_text())
        assert "parse_stats" not in dm and dm["classes"] == CLASSES and dm["n_task_samples"] == 18


def test_e2e_identical_test_sets_across_arms(e2e):
    for fold in FOLDS:
        sets = {arm: frozenset(pd.read_parquet(e2e.dirs[(arm, fold)] / "predictions.parquet").query("split == 'test'")
                               .sample_id) for arm in ARMS + BASE_ARMS}
        assert len(set(sets.values())) == 1, sets
    all_test = [s for f in FOLDS for s in pd.read_parquet(e2e.dirs[("functional", f)] / "predictions.parquet")
                .query("split == 'test'").sample_id]
    assert len(all_test) == 18 and len(set(all_test)) == 18  # every sample tested exactly once across folds


def test_e2e_baselines_fit_without_test_samples(e2e):
    # dmr_logistic calls region_matrix (fit on train+val, then val, then test) once per fold
    assert len(e2e.fit_calls) == 3 * len(FOLDS)
    for i, fold in enumerate(FOLDS):
        fit, val, test = e2e.fit_calls[3 * i : 3 * i + 3]
        pred = pd.read_parquet(e2e.dirs[("dmr_logistic", fold)] / "predictions.parquet")
        assert set(test) == set(pred[pred.split == "test"].sample_id)
        assert not set(fit) & set(test) and set(val) <= set(fit) and len(fit) + len(test) == 18
    rm = json.loads((e2e.dirs[("dmr_logistic", 0)] / "representation_manifest.json").read_text())
    assert rm["kind"] == "embedding_free_baseline" and rm["n_dmrs"] > 0 and rm["n_candidate_regions"] == 30


def test_e2e_representation_manifest(e2e):
    params = {}
    for arm in ARMS:
        rm = json.loads((e2e.dirs[(arm, 0)] / "representation_manifest.json").read_text())
        for k in ("raw_dim", "projected_dim", "explained_variance_ratio", "fit_universe_size", "pca_hash",
                  "source_representation_hash"):
            assert k in rm, (arm, k)
        params[arm] = rm["trainable_parameters"]["trainable"]
        if arm.startswith("functional"):
            assert rm["raw_dim"] == 16 and rm["projected_dim"] == 8 and rm["fit_universe_size"] == 300
            assert 0 < rm["explained_variance_ratio"] <= 1 and len(rm["pca_hash"]) == 64
            assert len(rm["source_representation_hash"]) == 64
    a = json.loads((e2e.dirs[("functional", 0)] / "representation_manifest.json").read_text())
    b = json.loads((e2e.dirs[("functional_shuffled", 0)] / "representation_manifest.json").read_text())
    assert a["pca_hash"] == b["pca_hash"] and a["source_representation_hash"] == b["source_representation_hash"]
    assert b["shuffle_loci_seed"] == 1 and a["shuffle_loci_seed"] is None
    assert params["functional"] == params["functional_shuffled"] == params["random"]
    assert params["methylation_only"] < params["functional"]


def test_e2e_attributions(e2e):
    for fold in FOLDS:
        path = e2e.dirs[("functional", fold)] / "attributions.parquet"
        assert path.exists()
        att = pd.read_parquet(path)
        assert {"sample_id", "true_class", "predicted_class", "chrom_code", "pos_hg38", "pos_hg19", "beta", "coverage",
                "pool_score", "contribution_pred", "contribution_true", "embedding_pc1", "embedding_pc4"} <= set(att.columns)
        pred = pd.read_parquet(e2e.dirs[("functional", fold)] / "predictions.parquet")
        assert set(att.sample_id) == set(pred[pred.split == "test"].sample_id)
        assert (att.groupby("sample_id").size() == 5).all()  # attribution_top_k from the attribution block
        assert (att.coverage > 0).all() and att.beta.notna().all()
        assert set(zip(att.chrom_code, att.pos_hg38)) <= {(int(k >> 32), int(k & 0xFFFFFFFF)) for k in e2e.loci}
        assert (att.pos_hg38 - att.pos_hg19 == 7).all()  # hg19 keys of the fixture are hg38 - 7
    for arm in ("methylation_only", "random", "functional_shuffled", "summary_only"):
        assert not (e2e.dirs[(arm, 0)] / "attributions.parquet").exists()


def test_e2e_selection_only(e2e):
    d = e2e.sel_dir
    for f in RUN_FILES + ["RUN_COMPLETE.json"]:
        assert (d / f).exists()
    pred = pd.read_parquet(d / "predictions.parquet")
    assert set(pred.split) == {"val"}
    m = json.loads((d / "metrics.json").read_text())
    assert not [k for k in m if k.startswith("test/")] and "val/all" in m
    assert "test" not in m["run"]["class_distribution"]
    assert not (d / "attributions.parquet").exists()


def test_e2e_summary(e2e):
    exp = e2e.exp
    exp_dir = e2e.dirs[("functional", 0)].parents[2]
    res = summarize(exp_dir, CLASSES, exp["seeds"], exp["folds"], [tuple(p) for p in exp["comparisons"]],
                    n_boot=20, n_perm=20)
    assert set(res["arms"].representation) == set(ARMS + BASE_ARMS)
    assert not res["incomplete"]
    assert (res["per_seed"].n_patients == 18).all()
    assert set(res["per_batch"].batch) == {"KRp1", "KRp2"}
    comp = res["comparisons"]
    assert set(comp.arm_a) == {"functional"}
    assert set(comp.arm_b) == {"methylation_only", "random", "functional_shuffled", "summary_only", "dmr_logistic"}


def test_e2e_unavailable_arm_fails_alone(e2e):
    with pytest.raises(RepresentationUnavailableError, match="missing_store"):
        br.run_beta(e2e.exp, "missing_store", SEED, 0, **QUIET)
    assert not (e2e.dirs[("functional", 0)].parents[2] / "missing_store" / f"seed_{SEED}" / "fold_0" /
                "RUN_COMPLETE.json").exists()
    for fold in FOLDS:  # the other arms are untouched and still complete
        assert (e2e.dirs[("functional", fold)] / "RUN_COMPLETE.json").exists()


def test_e2e_rerun_skips_complete(e2e):
    d = e2e.dirs[("summary_only", 0)]
    stamp = (d / "RUN_COMPLETE.json").stat().st_mtime_ns
    assert br.run_beta(e2e.exp, "summary_only", SEED, 0, **QUIET) == d
    assert (d / "RUN_COMPLETE.json").stat().st_mtime_ns == stamp
