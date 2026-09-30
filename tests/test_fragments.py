import json

import numpy as np
import pytest

from cfdna_origin.data.fragments import FragmentStore, finalize_dataset, heldout_locus_mask, write_sample
from cfdna_origin.data.loci import make_keys


def _k(*pos):
    return make_keys("chr1", np.asarray(pos))


@pytest.fixture()
def hand_store(tmp_path):
    """Sample A: f0 = 3 CpGs, f1 = 2 CpGs (invalid, min 3), f2 = 5 CpGs (truncated to max 4). Sample B: 1 fragment."""
    write_sample(tmp_path / "fragments" / "A", [0, 3, 5, 10],
                 _k(100, 110, 150, 200, 201, 300, 305, 320, 400, 401), [1, 0, 1, 1, 1, 0, 1, 1, 0, 1])
    write_sample(tmp_path / "fragments" / "B", [0, 3], _k(110, 150, 200), [0, 0, 1])
    finalize_dataset(tmp_path, ["A", "B"], {"dataset": "hand"})
    return FragmentStore(tmp_path, min_cpgs=3, max_cpgs=4)


@pytest.mark.parametrize("offs,keys,state,msg", [
    ([0, 2, 2], [1, 2], [0, 1], "empty"),                  # empty fragment
    ([0, 2], [1, 2], [0, 2], "0/1"),                       # bad state
    ([0, 3], [1, 2], [0, 1], "inconsistent"),              # offsets do not end at len(keys)
    ([1, 2], [1, 2], [0, 1], "inconsistent"),              # offsets do not start at 0
    ([0, 2], [1, 2], [0], "inconsistent"),                 # state length mismatch
])
def test_write_sample_validation(tmp_path, offs, keys, state, msg):
    with pytest.raises(ValueError, match=msg):
        write_sample(tmp_path / "s", offs, _k(*keys), state)


def test_finalize_dataset(hand_store, tmp_path):
    loci = np.load(tmp_path / "loci.npy")
    assert np.all(np.diff(loci) > 0)
    np.testing.assert_array_equal(loci, _k(100, 110, 150, 200, 201, 300, 305, 320, 400, 401))
    expected = {"A": _k(100, 110, 150, 200, 201, 300, 305, 320, 400, 401), "B": _k(110, 150, 200)}
    for s, keys in expected.items():
        assert not (tmp_path / "fragments" / s / "locus_key.npy").exists()  # dropped: keys = loci[locus_row]
        rows = np.load(tmp_path / "fragments" / s / "locus_row.npy")
        np.testing.assert_array_equal(loci[rows], keys)
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["n_loci"] == 10 and manifest["n_samples"] == 2 and manifest["dataset"] == "hand"


def test_filters_and_stats(hand_store):
    np.testing.assert_array_equal(hand_store.valid_fragments("A"), [0, 2])
    assert hand_store.n_fragments("A") == 3
    st = hand_store.sample_stats("A")
    assert st == {"number_of_reads": 3, "number_of_valid_reads": 2, "mean_CpGs_per_read": 3.5}  # (3 + min(5, 4)) / 2


def test_bag_shapes_mask_padding_geometry(hand_store):
    bag = hand_store.bag(["A", "B"], [np.array([0, 2]), np.array([0])])
    assert len(bag) == 3
    assert bag.rows.shape == bag.state.shape == bag.mask.shape == (3, 4) and bag.geom.shape == (3, 4, 3)
    np.testing.assert_array_equal(bag.sample, [0, 0, 1])
    np.testing.assert_array_equal(bag.mask, [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]])
    np.testing.assert_array_equal(bag.rows, [[0, 1, 2, 0], [5, 6, 7, 8], [1, 2, 3, 0]])  # f2 truncated at 400
    np.testing.assert_array_equal(bag.state, [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]])
    assert np.all(bag.geom[~bag.mask] == 0)
    g = bag.geom.astype(np.float64)
    np.testing.assert_allclose(g[0, :3, 0], np.log1p([0, 10, 50]) / 7, rtol=1e-6)   # offset from first CpG
    np.testing.assert_allclose(g[0, :3, 1], np.log1p([0, 10, 40]) / 7, rtol=1e-6)   # gap to previous
    np.testing.assert_allclose(g[0, :3, 2], [0, 0.5, 1], rtol=1e-6)                 # rank
    np.testing.assert_allclose(g[1, :, 0], np.log1p([0, 5, 20, 100]) / 7, rtol=1e-6)
    np.testing.assert_allclose(g[1, :, 1], np.log1p([0, 5, 15, 80]) / 7, rtol=1e-6)
    np.testing.assert_allclose(g[1, :, 2], [0, 1 / 3, 2 / 3, 1], rtol=1e-6)


def test_heldout_locus_mask_deterministic():
    loci = make_keys("chr3", np.arange(1, 20_001) * 7)
    a = heldout_locus_mask(loci, 0.2, seed=5)
    np.testing.assert_array_equal(a, heldout_locus_mask(loci, 0.2, seed=5))
    assert abs(a.mean() - 0.2) < 0.02
    assert not np.array_equal(a, heldout_locus_mask(loci, 0.2, seed=6))
    np.testing.assert_array_equal(heldout_locus_mask(loci[::2], 0.2, seed=5), a[::2])  # per-locus, not per-set
    assert not heldout_locus_mask(loci, 0.0, seed=5).any()


def test_valid_fragments_heldout_partition(tiny_dataset):
    root, samples = tiny_dataset
    store = FragmentStore(root, min_cpgs=3, max_cpgs=32)
    held = heldout_locus_mask(np.asarray(store.loci), 0.1, seed=0)
    n_only = 0
    for sid in samples.sample_id:
        valid = store.valid_fragments(sid)
        exc = store.valid_fragments(sid, held, "exclude")
        only = store.valid_fragments(sid, held, "only")
        assert not set(exc) & set(only)
        np.testing.assert_array_equal(np.sort(np.concatenate([exc, only])), valid)
        bag = store.bag([sid], [exc]) if len(exc) else None
        if bag is not None:
            assert not held[bag.rows[bag.mask]].any()
        n_only += len(only)
    assert n_only > 0


def test_finalize_is_incremental_after_locus_keys_are_dropped(tmp_path):
    from cfdna_origin.data.fragments import FragmentStore, finalize_dataset, write_sample
    from cfdna_origin.data.loci import make_keys

    a = make_keys("chr1", np.array([10, 20, 30, 40]))
    b = make_keys("chr2", np.array([5, 15, 25]))
    write_sample(tmp_path / "fragments" / "A", np.array([0, 4]), a, np.array([1, 0, 1, 0]))
    finalize_dataset(tmp_path, ["A"], {})
    assert not (tmp_path / "fragments" / "A" / "locus_key.npy").exists()  # compact store
    write_sample(tmp_path / "fragments" / "B", np.array([0, 3]), b, np.array([0, 0, 1]))
    finalize_dataset(tmp_path, ["A", "B"], {})
    store = FragmentStore(tmp_path, min_cpgs=1)
    np.testing.assert_array_equal(np.asarray(store.loci), np.sort(np.concatenate([a, b])))
    for sid, keys in (("A", a), ("B", b)):
        rows = np.load(tmp_path / "fragments" / sid / "locus_row.npy")
        np.testing.assert_array_equal(np.asarray(store.loci)[rows], keys)
    bag = store.bag(["A"], [np.array([0])])
    assert bag.geom[0, 1, 1] == pytest.approx(np.log1p(10) / 7.0)  # positions recovered from loci.npy


def test_filter_calls_drops_calls_and_emptied_fragments():
    from cfdna_origin.data.fragments import filter_calls

    off = np.array([0, 2, 3, 6])
    keys = _k(1, 2, 3, 4, 5, 6)
    keep = np.array([True, False, False, True, True, False])
    new_off, k, st, stats = filter_calls(off, keys, np.arange(6) % 2, keep)
    np.testing.assert_array_equal(new_off, [0, 1, 3])
    np.testing.assert_array_equal(k, _k(1, 4, 5))
    assert stats == {"calls_dropped": 3, "fragments_emptied": 1}
