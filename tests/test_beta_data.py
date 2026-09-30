"""Processed-beta data path: methratio parsing, GEO tar access, liftover/CpG/collision filtering, BetaData, splits.

Synthetic, tiny, in-memory / tmp_path fixtures only.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest

from cfdna_origin.data.beta import BetaData
from cfdna_origin.data.datasets import gse149438_beta as ds
from cfdna_origin.data.liftover import Chain
from cfdna_origin.data.loci import make_keys
from cfdna_origin.data.splits import SplitError, check_no_leakage, confounded_classes, make_splits
from cfdna_origin.experiments.beta_runner import split_spec
from cfdna_origin.representations.hdf5 import HDF5LocusEmbeddingStore
from conftest import write_h5

HEADER = "\t".join(ds.COLUMNS)


def _table(rows) -> str:
    """rows: (chr, pos, strand, context, C, CT)."""
    lines = [HEADER]
    for chrom, pos, strand, ctx, c, ct in rows:
        ratio = c / ct if ct else 0.0
        lines.append(f"{chrom}\t{pos}\t{strand}\t{ctx}\t{ratio:.3f}\t{ct}.00\t{c}\t{ct}\t{ct}\t{ct}\t0.1\t0.9")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- methratio parsing
def test_parse_methratio_filters_and_keys():
    text = _table([
        ("chr2", 500, "+", "CG", 3, 4),
        ("chr1", 200, "+", "CG", 5, 10),
        ("chr1", 100, "+", "CG", 0, 4),
        ("chr1", 150, "+", "CHG", 2, 9),     # non-CG
        ("chrX", 100, "+", "CG", 2, 9),      # non-autosome
        ("chrM", 10, "+", "CG", 2, 9),       # non-autosome
        ("chr3", 300, "+", "CG", 1, 3),      # coverage < 4
    ])
    keys, c, ct, st = ds.parse_methratio(io.StringIO(text), min_coverage=4)
    np.testing.assert_array_equal(keys, [(1 << 32) | 100, (1 << 32) | 200, (2 << 32) | 500])
    np.testing.assert_array_equal(keys, make_keys(["chr1", "chr1", "chr2"], [100, 200, 500]))
    np.testing.assert_array_equal(c, [0, 5, 3])
    np.testing.assert_array_equal(ct, [4, 10, 4])
    assert st == {"rows": 7, "dropped_non_cg": 1, "dropped_other_contig": 2, "dropped_low_coverage": 1, "kept": 3}


def test_parse_methratio_coverage_threshold_inclusive():
    text = _table([("chr1", 100, "+", "CG", 1, 4), ("chr1", 200, "+", "CG", 1, 5)])
    keys, _, _, st = ds.parse_methratio(io.StringIO(text), min_coverage=5)
    np.testing.assert_array_equal(keys, make_keys("chr1", [200]))
    assert st["dropped_low_coverage"] == 1


@pytest.mark.parametrize("rows, msg", [
    ([("chr1", 100, "+", "CG", 1, 4), ("chr1", 100, "+", "CG", 2, 5)], "duplicated"),
    ([("chr1", 100, "+", "CG", 6, 5)], "C_count > CT_count"),
    ([("chr1", 100, "-", "CG", 1, 5)], "strand"),
])
def test_parse_methratio_raises(rows, msg):
    with pytest.raises(ValueError, match=msg):
        ds.parse_methratio(io.StringIO(_table(rows)), min_coverage=4)


def test_parse_methratio_ignores_minus_strand_on_dropped_rows():
    # a '-' row that is filtered out anyway (non-CG) does not trigger the strand check
    text = _table([("chr1", 100, "-", "CHH", 1, 5), ("chr1", 200, "+", "CG", 1, 5)])
    keys, _, _, _ = ds.parse_methratio(io.StringIO(text), min_coverage=4)
    assert len(keys) == 1


# ---------------------------------------------------------------- GEO tar access
@pytest.fixture()
def geo_tar(tmp_path):
    path = tmp_path / "GSE_RAW.tar"
    tables = {"GSM1_KRp1-1_bsmap_meth.txt.gz": _table([("chr1", 100, "+", "CG", 2, 4), ("chr1", 90, "+", "CG", 4, 8)]),
              "GSM2_KRp2-5_bsmap_meth.txt.gz": _table([("chr5", 7, "+", "CG", 1, 6)])}
    with tarfile.open(path, "w") as tar:
        for name, text in {**tables, "README.txt": "unrelated"}.items():
            raw = gzip.compress(text.encode()) if name.endswith(".gz") else text.encode()
            info = tarfile.TarInfo(name); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return path


def test_list_members_and_read_member(geo_tar):
    members = ds.list_members(geo_tar)
    assert members == {"GSM1": "GSM1_KRp1-1_bsmap_meth.txt.gz", "GSM2": "GSM2_KRp2-5_bsmap_meth.txt.gz"}
    keys, c, ct, st = ds.read_member(geo_tar, members["GSM1"], 4)
    np.testing.assert_array_equal(keys, make_keys("chr1", [90, 100]))
    np.testing.assert_array_equal(c, [4, 2]); np.testing.assert_array_equal(ct, [8, 4])
    assert st["kept"] == 2
    keys2, _, _, _ = ds.read_member(geo_tar, members["GSM2"], 4)
    np.testing.assert_array_equal(keys2, make_keys("chr5", [7]))


def test_build_samples_table(geo_tar):
    members = ds.list_members(geo_tar)
    meta = pd.DataFrame({"sample_id": ["GSM1", "GSM2"], "batch": ["KRp1", "KRp2"]})
    out = ds.build_samples_table(meta, members)
    assert out.source_file.tolist() == ["GSM1_KRp1-1_bsmap_meth.txt.gz", "GSM2_KRp2-5_bsmap_meth.txt.gz"]
    with pytest.raises(FileNotFoundError, match="no methratio file"):
        ds.build_samples_table(pd.DataFrame({"sample_id": ["GSM1", "GSM2", "GSM3"],
                                             "batch": ["KRp1", "KRp2", "KRp1"]}), members)
    with pytest.raises(ValueError, match="without metadata"):
        ds.build_samples_table(meta.iloc[:1], members)
    with pytest.raises(ValueError, match="batch in the file name"):
        ds.build_samples_table(meta.assign(batch=["KRp1", "KRp1"]), members)


# ---------------------------------------------------------------- collisions / summaries
def test_resolve_collisions_many_to_one():
    target = np.array([10, 20, 10, 30, 20, 10, 40, 50])
    ok = np.array([True, True, True, True, False, True, True, False])
    # 10 is the target of three valid sources -> all three dropped; 20 is shared only with an invalid source -> kept;
    # 50 is invalid anyway
    keep, n = ds.resolve_collisions(target, ok)
    np.testing.assert_array_equal(keep, [False, True, False, True, False, False, True, False])
    assert n == 3


def test_sample_summary_values():
    beta = np.array([0.5, np.nan, 1.0, 0.0], np.float32)
    cov = np.array([4, 0, 8, 6])
    s = ds.sample_summary(beta, cov)
    assert s["n_observed_loci"] == 3
    assert s["missing_fraction"] == pytest.approx(0.25)
    assert s["mean_beta"] == pytest.approx(0.5)
    assert s["beta_variance"] == pytest.approx(np.var([0.5, 1.0, 0.0]))
    assert s["mean_coverage"] == pytest.approx(6.0) and s["median_coverage"] == pytest.approx(6.0)
    empty = ds.sample_summary(np.full(2, np.nan), np.zeros(2))
    assert empty["n_observed_loci"] == 0 and empty["missing_fraction"] == 1.0 and np.isnan(empty["mean_beta"])


# ---------------------------------------------------------------- liftover + CpG check as used by prepare
# chain 1: hg19 chr1 [100, 200) -> chr1 [1000, 1100); chain 2: hg19 chr2 [0, 50) -> chr1 [1050, 1100) (collides)
CHAIN = """\
chain 1000 chr1 10000 + 100 200 chr1 20000 + 1000 1100 1
100

chain 900 chr2 10000 + 0 50 chr1 20000 + 1050 1100 2
50
"""


def _prepare_filter(chain, union, ref, store):
    """The filtering sequence of scripts/prepare_gse149438_beta.py (liftover -> CpG -> collisions -> store)."""
    target, ok_nocheck, st_nocheck = chain.lift(union)
    _, ok, st = chain.lift(union, ref)
    ok, n_collision = ds.resolve_collisions(target, ok)
    in_stores = ok.copy()
    in_stores[ok] &= store.covers(target[ok])
    return target, {"n_lifted": int(ok_nocheck.sum()), "n_failed": st_nocheck["unmapped"],
                    "n_non_CpG": st["not_cpg_in_target"], "n_collision": n_collision,
                    "n_valid": int(ok.sum()), "n_final": int(in_stores.sum())}, in_stores


def test_liftover_cpg_check_integration(tmp_path):
    path = tmp_path / "toy.over.chain.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(CHAIN)
    chain = Chain(path)
    # hg19 sources: chr1:101 -> 1001 (CpG), chr1:111 -> 1011 (NOT a CpG in hg38), chr1:121 -> 1021 (CpG, not in store),
    # chr1:161 -> 1061 and chr2:11 -> 1061 (many-to-one), chr1:5 unmapped
    union = np.sort(np.concatenate([make_keys("chr1", [5, 101, 111, 121, 161]), make_keys("chr2", [11])]))
    ref = make_keys("chr1", [1001, 1021, 1061, 1500])
    store_keys = make_keys("chr1", [1001, 1061, 1500])
    store = HDF5LocusEmbeddingStore(write_h5(tmp_path / "s.h5", store_keys, np.zeros((3, 2), np.float32)),
                                    name="f", normalization="none")
    target, stats, final = _prepare_filter(chain, union, ref, store)
    assert stats == {"n_lifted": 5, "n_failed": 1, "n_non_CpG": 1, "n_collision": 2, "n_valid": 2, "n_final": 1}
    np.testing.assert_array_equal(union[final], make_keys("chr1", [101]))
    np.testing.assert_array_equal(target[final], make_keys("chr1", [1001]))


# ---------------------------------------------------------------- BetaData
def write_processed(root, loci, beta, cov, samples: pd.DataFrame, manifest: dict | None = None):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "loci.npy", np.asarray(loci, np.int64))
    np.save(root / "loci_hg19.npy", np.asarray(loci, np.int64) - 7)
    np.save(root / "beta.npy", np.asarray(beta, np.float16))
    np.save(root / "coverage.npy", np.asarray(cov, np.uint16))
    rows = [ds.sample_summary(beta[i].astype(np.float32), cov[i].astype(np.int64)) for i in range(len(samples))]
    samples = pd.concat([samples.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    samples.to_parquet(root / "samples.parquet", index=False)
    (root / "dataset_manifest.json").write_text(json.dumps(manifest or {"dataset": "tiny_beta", "n_loci": len(loci)}))
    return samples


@pytest.fixture()
def tiny_beta(tmp_path):
    loci = make_keys("chr1", [10, 20, 30, 40, 50])
    beta = np.array([[0.25, np.nan, 1.0, 0.5, np.nan], [0.0, 0.75, np.nan, np.nan, 1.0]], np.float32)
    cov = np.array([[4, 0, 8, 6, 0], [5, 9, 0, 0, 4]])
    samples = pd.DataFrame({"sample_id": ["A", "B"], "patient_id": ["pA", "pB"], "diagnosis": ["Normal", "CRC"],
                            "batch": ["KRp1", "KRp2"]})
    write_processed(tmp_path / "proc", loci, beta, cov, samples)
    return tmp_path / "proc", loci


@pytest.mark.parametrize("in_memory", [True, False])
def test_betadata_observed_tokens(tiny_beta, in_memory):
    root, loci = tiny_beta
    d = BetaData(root, in_memory=in_memory)
    np.testing.assert_array_equal(d.loci, loci)
    np.testing.assert_array_equal(d.observed("A"), [0, 2, 3])
    np.testing.assert_array_equal(d.observed("B"), [0, 1, 4])
    cols, b, c = d.tokens("A", d.observed("A"))
    np.testing.assert_array_equal(cols, [0, 2, 3])
    np.testing.assert_allclose(b, [0.25, 1.0, 0.5]); np.testing.assert_allclose(c, [4, 8, 6])
    assert b.dtype == np.float32 and c.dtype == np.float32 and not np.isnan(b).any()
    assert len(d.manifest_sha256) == 64


def test_betadata_summary_features(tiny_beta):
    d = BetaData(tiny_beta[0])
    f = d.summary_features(["B", "A"])
    assert list(f.index) == ["B", "A"]
    assert list(f.columns) == ["n_observed_loci", "mean_beta", "beta_variance", "mean_coverage", "median_coverage",
                               "missing_fraction"]
    assert f.loc["A", "n_observed_loci"] == 3 and f.loc["B", "mean_coverage"] == pytest.approx(6.0)
    assert f.loc["A", "missing_fraction"] == pytest.approx(0.4)


def test_betadata_shape_validation(tiny_beta):
    root, _ = tiny_beta
    np.save(root / "coverage.npy", np.zeros((2, 4), np.uint16))
    with pytest.raises(ValueError, match="shapes"):
        BetaData(root)
    (root / "dataset_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        BetaData(root)


# ---------------------------------------------------------------- splits: cross_batch, confounding, seed_from_run
def _batch_samples(per_class_batch):
    """per_class_batch: {(label, batch): n_patients}."""
    rows = []
    for (label, batch), n in per_class_batch.items():
        for p in range(n):
            sid = f"S_{label}_{batch}_{p}"
            rows.append({"sample_id": sid, "patient_id": f"P_{sid}", "label": label, "batch": batch})
    return pd.DataFrame(rows)


def test_cross_batch_folds():
    s = _batch_samples({(c, b): 6 for c in ("A", "B") for b in ("KRp1", "KRp2")})
    sp = make_splits(s, strategy="cross_batch", seed=3, val_fraction=0.25)
    check_no_leakage(sp)
    sp = sp.merge(s[["sample_id", "batch"]])
    assert sorted(sp.fold.unique()) == [0, 1]
    for fold, test_batch in enumerate(["KRp1", "KRp2"]):
        g = sp[sp.fold == fold]
        assert set(g[g.split == "test"].batch) == {test_batch}
        assert (g[g.batch == test_batch].split == "test").all()
        assert set(g[g.split == "val"].batch) == {"KRp2" if test_batch == "KRp1" else "KRp1"}
        assert set(g[g.split == "val"].label) == {"A", "B"}


def test_cross_batch_class_absent_raises():
    s = _batch_samples({("A", "KRp1"): 5, ("A", "KRp2"): 5, ("B", "KRp1"): 5, ("C", "KRp2"): 5})
    with pytest.raises(SplitError, match="every class in every batch"):
        make_splits(s, strategy="cross_batch", seed=0)


def test_confounded_classes():
    s = _batch_samples({("A", "KRp1"): 3, ("A", "KRp2"): 3, ("CRC", "KRp2"): 4, ("B", "KRp1"): 2})
    assert confounded_classes(s) == {"CRC": "KRp2", "B": "KRp1"}
    assert confounded_classes(_batch_samples({("A", "KRp1"): 2, ("A", "KRp2"): 2})) == {}


def test_split_spec_seed_from_run_and_protocol_override(tiny_samples):
    exp = {"dataset": {"split": {"strategy": "stratified_kfold", "n_folds": 5, "seed": 1, "seed_from_run": True,
                                 "stratify_by": ["label", "batch"]}},
           "protocol": {"name": "all_classes_original"}}
    spec = split_spec(exp, 42)
    assert spec["seed"] == 42 and "seed_from_run" not in spec
    assert exp["dataset"]["split"]["seed_from_run"] is True  # the experiment dict is not mutated
    kw = {k: spec[k] for k in ("strategy", "seed", "n_folds", "stratify_by")}
    a, b, c = (make_splits(tiny_samples, **kw), make_splits(tiny_samples, **{**kw, "seed": 42}),
               make_splits(tiny_samples, **{**kw, "seed": 43}))
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    exp["dataset"]["split"]["seed_from_run"] = False
    assert split_spec(exp, 42)["seed"] == 1
    exp["protocol"]["split"] = {"strategy": "cross_batch"}
    assert split_spec(exp, 42)["strategy"] == "cross_batch"
