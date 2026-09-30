import json

import numpy as np
import pytest

from cfdna_origin.data.loci import make_keys
from cfdna_origin.representations.base import NoLocusStore, RepresentationCoverageError
from cfdna_origin.representations.controls import PositionOnlyStore, RandomLocusEmbeddingStore
from cfdna_origin.representations.hdf5 import HDF5LocusEmbeddingStore
from cfdna_origin.representations.registry import (
    RepresentationUnavailableError,
    build_store,
    load_materialized,
    materialize,
)
from conftest import write_h5

KEYS = np.sort(np.concatenate([make_keys("chr1", np.arange(1, 201) * 50), make_keys("chr2", np.arange(1, 101) * 70)]))
DIM = 12


@pytest.fixture()
def emb():
    return np.random.default_rng(0).normal(3.0, 2.0, size=(len(KEYS), DIM)).astype(np.float32)


@pytest.fixture()
def sorted_h5(tmp_path, emb):
    return write_h5(tmp_path / "sorted.h5", KEYS, emb, chunks=(32, DIM), attrs={"reference_build": "GRCh38"})


@pytest.fixture()
def unsorted_h5(tmp_path, emb):
    perm = np.random.default_rng(1).permutation(len(KEYS))
    return write_h5(tmp_path / "unsorted.h5", KEYS[perm], emb[perm], chunks=(32, DIM))


@pytest.mark.parametrize("which", ["sorted_h5", "unsorted_h5"])
def test_hdf5_lookup_rows(request, emb, which):
    store = HDF5LocusEmbeddingStore(request.getfixturevalue(which), name="f", normalization="none")
    pick = np.array([299, 0, 17, 150, 17])  # arbitrary order, repeated key
    np.testing.assert_allclose(store.get_embeddings(KEYS[pick]), emb[pick], rtol=1e-6)
    row = np.searchsorted(KEYS, make_keys("chr2", [210]))[0]
    np.testing.assert_allclose(store.get_embedding("chr2", 210), emb[row])
    assert store.dim == DIM and store.covers(KEYS).all()


@pytest.mark.parametrize("which", ["sorted_h5", "unsorted_h5"])
def test_hdf5_missing_locus_raises(request, which):
    store = HDF5LocusEmbeddingStore(request.getfixturevalue(which), name="f", normalization="none")
    missing = make_keys("chr1", [51])
    assert not store.covers(np.concatenate([KEYS[:3], missing]))[-1]
    with pytest.raises(RepresentationCoverageError):
        store.get_embeddings(np.concatenate([KEYS[:3], missing]))
    with pytest.raises(RepresentationCoverageError):
        store.get_embeddings(make_keys("chrX", [10**8]))  # beyond the last key


def test_hdf5_duplicate_keys_rejected(tmp_path):
    path = write_h5(tmp_path / "dup.h5", KEYS[[3, 1, 3]], np.zeros((3, 2), np.float32))
    with pytest.raises(RepresentationCoverageError, match="duplicate"):
        HDF5LocusEmbeddingStore(path, name="d", normalization="none")


def test_hdf5_manifest(sorted_h5):
    m = HDF5LocusEmbeddingStore(sorted_h5, name="functional", normalization="none").manifest.to_dict()
    assert m["name"] == "functional" and m["kind"] == "hdf5" and m["dim"] == DIM
    assert m["source"] == str(sorted_h5) and m["genome_build"] == "GRCh38"
    assert m["key_namespace"] == "locus_key_chromcode<<32|pos_v1" and m["n_universe_loci"] == len(KEYS)
    assert len(m["provenance"]["store_fingerprint_sha256"]) == 64
    assert m["normalization"]["data_derived"] is False


def test_store_sample_standardize(sorted_h5):
    store = HDF5LocusEmbeddingStore(sorted_h5, name="f", normalization="store_sample_standardize",
                                    norm_sample_chunks=64)
    x = store.get_embeddings(KEYS)  # all chunks sampled -> exact standardisation of the store
    np.testing.assert_allclose(x.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(x.std(0), 1, atol=1e-3)
    assert store.manifest.normalization["type"] == "store_sample_standardize"
    assert store.manifest.normalization["data_derived"] is False
    with pytest.raises(ValueError):
        HDF5LocusEmbeddingStore(sorted_h5, name="f", normalization="zscore_on_dataset")


def test_random_store():
    store = RandomLocusEmbeddingStore(dim=32, seed=0)
    keys = make_keys("chr5", np.arange(1, 3001) * 11)
    full = store.get_embeddings(keys)
    perm = np.random.default_rng(0).permutation(len(keys))[:500]
    np.testing.assert_array_equal(store.get_embeddings(keys[perm]), full[perm])  # batch/order independent
    np.testing.assert_array_equal(RandomLocusEmbeddingStore(dim=32, seed=0).get_embeddings(keys[:5]), full[:5])
    assert not np.allclose(RandomLocusEmbeddingStore(dim=32, seed=1).get_embeddings(keys[:5]), full[:5])
    assert abs(full.mean()) < 0.03 and abs(full.std() - 1) < 0.03
    assert store.manifest.normalization["data_derived"] is False
    with pytest.raises(RepresentationCoverageError):
        store.get_embeddings(np.array([25 << 32 | 5]))  # chrom code 25 does not exist


def test_position_only_store():
    store = PositionOnlyStore(dim=64)
    x = store.get_embeddings(make_keys("chr3", [1_000_000, 1_000_002, 90_000_000]))
    cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)  # noqa: E731
    assert cos(x[0], x[1]) > cos(x[0], x[2])
    np.testing.assert_array_equal(x[:, :24].argmax(1), [2, 2, 2])
    assert x[:, :24].sum(1).tolist() == [1, 1, 1]
    assert store.get_embedding("chrY", 5)[23] == 1
    with pytest.raises(ValueError):
        PositionOnlyStore(dim=25)


def test_no_locus_store():
    store = NoLocusStore()
    assert store.dim == 0 and store.get_embeddings(KEYS[:4]).shape == (4, 0)


def test_build_store_each_kind(tmp_path, sorted_h5):
    paths = {"representations_root": tmp_path}
    assert isinstance(build_store({"name": "m", "kind": "none"}, paths), NoLocusStore)
    assert build_store({"name": "r", "kind": "random", "dim": 8, "seed": 2}, paths).dim == 8
    assert build_store({"name": "p", "kind": "position", "dim": 30}, paths).dim == 30
    h = build_store({"name": "f", "kind": "hdf5", "path": "{representations_root}/sorted.h5", "normalization": "none"},
                    paths)
    assert isinstance(h, HDF5LocusEmbeddingStore) and h.dim == DIM
    with pytest.raises(RepresentationUnavailableError, match="not found"):
        build_store({"name": "x", "kind": "hdf5", "path": "{representations_root}/absent.h5"}, paths)
    with pytest.raises(ValueError):
        build_store({"name": "x", "kind": "magic"}, paths)


LOCI = KEYS[::3]


def test_materialize_pca(tmp_path, sorted_h5):
    store = HDF5LocusEmbeddingStore(sorted_h5, name="f", normalization="none")
    m = materialize(store, LOCI, tmp_path / "pca", dataset_manifest_sha="abc",
                    projection={"type": "pca", "dim": 4, "fit_loci": 300, "seed": 0})
    table, m2 = load_materialized(tmp_path / "pca", len(LOCI), "abc")
    assert table.shape == (len(LOCI), 4) and table.dtype == np.float16
    assert m["dim_materialized"] == 4 and m2["projection"]["dim"] == 4
    assert 0 < m["projection"]["explained_variance_ratio"] <= 1
    assert m["projection"]["data_derived"] is False and m["normalization"]["data_derived"] is False
    assert json.loads((tmp_path / "pca" / "manifest.json").read_text())["materialization"]["n_loci"] == len(LOCI)


def test_materialize_none_and_shuffle(tmp_path, sorted_h5, emb):
    store = HDF5LocusEmbeddingStore(sorted_h5, name="f", normalization="none")
    materialize(store, LOCI, tmp_path / "raw", dataset_manifest_sha="abc", projection={"type": "none"})
    raw, m = load_materialized(tmp_path / "raw", len(LOCI), "abc")
    np.testing.assert_allclose(raw, emb[::3].astype(np.float16))
    assert m["projection"] == {"type": "none"} and m["dim_materialized"] == DIM
    materialize(store, LOCI, tmp_path / "shuf", dataset_manifest_sha="abc", shuffle_loci_seed=1)
    shuf, m = load_materialized(tmp_path / "shuf", len(LOCI), "abc")
    assert m["shuffle_loci_seed"] == 1
    assert not np.array_equal(shuf, raw)
    order = lambda a: a[np.lexsort(a.T[::-1])]  # noqa: E731  (sort rows to compare multisets)
    np.testing.assert_array_equal(order(np.asarray(shuf)), order(np.asarray(raw)))


def test_materialize_coverage_and_mismatch(tmp_path, sorted_h5):
    store = HDF5LocusEmbeddingStore(sorted_h5, name="f", normalization="none")
    with pytest.raises(RepresentationCoverageError):
        materialize(store, np.sort(np.concatenate([LOCI, make_keys("chr1", [51])])), tmp_path / "bad",
                    dataset_manifest_sha="abc")
    with pytest.raises(ValueError, match="strictly increasing"):
        materialize(store, LOCI[::-1], tmp_path / "bad", dataset_manifest_sha="abc")
    materialize(store, LOCI, tmp_path / "ok", dataset_manifest_sha="abc")
    with pytest.raises(ValueError, match="different dataset"):
        load_materialized(tmp_path / "ok", len(LOCI), "other_sha")
    with pytest.raises(ValueError, match="different dataset"):
        load_materialized(tmp_path / "ok", len(LOCI) + 1, "abc")
    with pytest.raises(RepresentationUnavailableError):
        load_materialized(tmp_path / "never", len(LOCI), "abc")


def test_materialize_controls_with_pca(tmp_path):
    for store in (RandomLocusEmbeddingStore(dim=16), PositionOnlyStore(dim=32)):
        m = materialize(store, LOCI, tmp_path / store.name, dataset_manifest_sha="s",
                        projection={"type": "pca", "dim": 8, "fit_loci": 500, "seed": 0})
        assert m["dim_materialized"] == 8 and m["projection"]["data_derived"] is False
    m = materialize(NoLocusStore(), LOCI, tmp_path / "none", dataset_manifest_sha="s",
                    projection={"type": "pca", "dim": 8})
    assert m["dim_materialized"] == 0 and m["projection"] == {"type": "none"}
