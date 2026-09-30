"""CpG-set classifier and embedding-free baselines of the processed-beta pilot (tiny synthetic inputs, CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cfdna_origin.data.beta import BetaData
from cfdna_origin.data.loci import make_keys
from cfdna_origin.models.classical import (
    DMRClassifier,
    SummaryClassifier,
    candidate_regions,
    region_matrix,
    select_dmrs,
)
from cfdna_origin.models.cpgset import CpGSetClassifier
from cfdna_origin.training import beta_trainer
from cfdna_origin.training.trainer import LocusTable
from test_beta_data import write_processed

ARCHS = ["mean", "deepsets", "pma"]
C = 3


def _cfg(arch, **kw):
    return {"arch": arch, "d_model": 8, "hidden": 16, "n_heads": 4, "dropout": 0.1, "head_dropout": 0.25, **kw}


def _tokens(n, locus_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(n, locus_dim, generator=g) if locus_dim else None
    return emb, torch.rand(n, generator=g), torch.randint(1, 50, (n,), generator=g).float()


# ---------------------------------------------------------------- CpGSetClassifier
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("locus_dim", [0, 8])
def test_forward_shapes_and_permutation_invariance(arch, locus_dim):
    torch.manual_seed(0)
    m = CpGSetClassifier(locus_dim, C, _cfg(arch)).eval()
    emb, b, c = _tokens(30, locus_dim)
    seg = torch.tensor([0] * 10 + [1] * 20)
    logits, w = m(emb, b, c, seg, 2)
    assert logits.shape == (2, C) and w.shape[0] == 30
    # permuting the tokens of sample 1 does not change its logits
    perm = torch.cat([torch.arange(10), 10 + torch.randperm(20, generator=torch.Generator().manual_seed(1))])
    logits_p, _ = m(emb[perm] if emb is not None else None, b[perm], c[perm], seg, 2)
    torch.testing.assert_close(logits, logits_p, atol=1e-5, rtol=1e-5)
    # segments are independent: sample 0 alone gives the same logits
    alone, _ = m(emb[:10] if emb is not None else None, b[:10], c[:10], torch.zeros(10, dtype=torch.long), 1)
    torch.testing.assert_close(alone[0], logits[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("locus_dim", [0, 8])
def test_predict_streaming_equals_forward(arch, locus_dim):
    torch.manual_seed(0)
    m = CpGSetClassifier(locus_dim, C, _cfg(arch)).eval()
    emb, b, c = _tokens(53, locus_dim)
    full, _ = m(emb, b, c, torch.zeros(53, dtype=torch.long), 1)
    chunks = [((emb[s : s + 10] if emb is not None else None), b[s : s + 10], c[s : s + 10]) for s in range(0, 53, 10)]
    torch.testing.assert_close(m.predict_streaming(iter(chunks)), full[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_token_scores_shapes(arch):
    m = CpGSetClassifier(8, C, _cfg(arch)).eval()
    emb, b, c = _tokens(17, 8)
    s, solo = m.token_scores(emb, b, c)
    assert s.shape == (17,) and solo.shape == (17, C)
    assert torch.isfinite(s).all() and torch.isfinite(solo).all()


@pytest.mark.parametrize("arch", ARCHS)
def test_trainable_parameter_count_depends_only_on_locus_dim(arch):
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
    torch.manual_seed(0); a = CpGSetClassifier(8, C, _cfg(arch))
    torch.manual_seed(1); b = CpGSetClassifier(8, C, _cfg(arch))
    assert count(a) == count(b)
    assert count(CpGSetClassifier(0, C, _cfg(arch))) < count(a)


@pytest.mark.parametrize("arch", ARCHS)
def test_use_coverage_false_ignores_coverage(arch):
    m = CpGSetClassifier(8, C, _cfg(arch, use_coverage=False)).eval()
    assert m.value.in_features == 1
    emb, b, c = _tokens(20, 8)
    seg = torch.zeros(20, dtype=torch.long)
    torch.testing.assert_close(m(emb, b, c, seg, 1)[0], m(emb, b, c * 7 + 3, seg, 1)[0])
    m2 = CpGSetClassifier(8, C, _cfg(arch)).eval()
    assert not torch.allclose(m2(emb, b, c, seg, 1)[0], m2(emb, b, c * 7 + 3, seg, 1)[0])


def test_unknown_arch():
    with pytest.raises(ValueError, match="unknown arch"):
        CpGSetClassifier(8, C, _cfg("transformer"))


# ---------------------------------------------------------------- frozen locus table under training
def _tiny_processed(root, n_per_class=4, n_loci=60, seed=0):
    rng = np.random.default_rng(seed)
    loci = make_keys("chr1", np.arange(1, n_loci + 1) * 10)
    rows, betas, covs = [], [], []
    for k, diag in enumerate(["Normal", "CRC", "GC"]):
        for p in range(n_per_class):
            cov = rng.integers(4, 30, n_loci) * (rng.random(n_loci) > 0.3)
            beta = np.where(np.arange(n_loci) // 20 == k, 0.9, 0.1) + rng.normal(0, 0.05, n_loci)
            beta = np.where(cov > 0, np.clip(beta, 0, 1), np.nan)
            betas.append(beta); covs.append(cov)
            rows.append({"sample_id": f"S{k}_{p}", "patient_id": f"P{k}_{p}", "diagnosis": diag, "label_idx": k,
                         "batch": f"KRp{p % 2 + 1}"})
    samples = write_processed(root, loci, np.array(betas), np.array(covs), pd.DataFrame(rows))
    return BetaData(root), samples


def test_training_never_updates_frozen_table(tmp_path):
    data, samples = _tiny_processed(tmp_path / "p")
    table_np = np.random.default_rng(0).normal(size=(len(data.loci), 8)).astype(np.float16)
    before = table_np.copy()
    table = LocusTable(table_np, "cpu")
    model = CpGSetClassifier(8, C, _cfg("pma"))
    assert not any(p.shape == table_np.shape for p in model.parameters())
    ids = samples.sample_id.tolist()
    y = samples.label_idx.to_numpy()
    cfg = {"samples_per_step": 3, "tokens_per_sample": 16, "bags_per_sample_per_epoch": 1, "max_epochs": 2,
           "patience": 5, "lr": 1e-2, "weight_decay": 0.0, "val_max_tokens": 20, "eval_chunk": 7, "eval_seed": 0}
    res = beta_trainer.train(model, data, table, train_ids=ids, train_labels=y, val_ids=ids[::2], val_labels=y[::2],
                             n_classes=C, cfg=cfg, seed=0, device="cpu", log=lambda *_: None)
    assert res["best_state"] is not None and len(res["history"]) == 2
    np.testing.assert_array_equal(table_np, before)
    # chunked prediction over all observed CpGs equals a single-chunk prediction
    a = beta_trainer.predict(model, data, table, ids[:3], max_tokens=None, chunk=7, seed=0, device="cpu")
    b = beta_trainer.predict(model, data, table, ids[:3], max_tokens=None, chunk=10_000, seed=0, device="cpu")
    np.testing.assert_allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------- classical baselines
def test_candidate_regions():
    loci = np.concatenate([make_keys("chr1", [100, 150, 250, 251, 600]), make_keys("chr2", [601, 650])])
    np.testing.assert_array_equal(candidate_regions(loci, max_gap=100), [0, 0, 0, 0, 1, 2, 2])
    np.testing.assert_array_equal(candidate_regions(loci, max_gap=99), [0, 0, 1, 1, 2, 3, 3])


class _FakeData:
    def __init__(self, beta, cov):
        self.beta, self.cov, self.ids = np.asarray(beta, np.float32), np.asarray(cov, np.float32), {"a": 0, "b": 1}

    def observed(self, sid):
        return np.flatnonzero(self.cov[self.ids[sid]] > 0)

    def tokens(self, sid, cols):
        r = self.ids[sid]
        return cols, self.beta[r, cols], self.cov[r, cols]


def test_region_matrix_coverage_weighted():
    region_of = np.array([0, 0, 1, 1, 2])
    data = _FakeData([[0.5, 1.0, 0.2, np.nan, np.nan], [0.0, np.nan, np.nan, np.nan, 0.4]],
                     [[2, 6, 5, 0, 0], [4, 0, 0, 0, 10]])
    X, sizes = region_matrix(data, ["a", "b"], region_of)
    np.testing.assert_allclose(X[0, :2], [(0.5 * 2 + 1.0 * 6) / 8, 0.2])
    assert np.isnan(X[0, 2]) and np.isnan(X[1, 1])
    np.testing.assert_allclose(X[1, [0, 2]], [0.0, 0.4])
    np.testing.assert_array_equal(sizes, [2, 2, 1])


def _dmr_fixture(seed=0):
    """12 samples x 30 regions; rows 0-5 are pure noise, rows 6-11 carry class signal in regions 0-5."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1, 2] * 4)
    X = rng.normal(0.5, 0.01, (12, 30))
    for c in range(3):
        rows = np.flatnonzero(y == c)
        rows = rows[rows >= 6]
        X[np.ix_(rows, [2 * c, 2 * c + 1])] += 0.4
    return X, y


def test_select_dmrs_uses_only_given_rows():
    X, y = _dmr_fixture()
    kw = dict(n_per_class=2, min_delta=0.1, min_observed_frac=0.5)
    sel_noise, info = select_dmrs(X[:6], y[:6], 3, **kw)
    assert len(sel_noise) == 0 and info["n_dmrs"] == 0
    sel_all, _ = select_dmrs(X, y, 3, **kw)
    assert len(sel_all) and set(sel_all) <= set(range(6))
    sel_signal, info = select_dmrs(X[6:], y[6:], 3, **kw)
    np.testing.assert_array_equal(sel_signal, np.arange(6))
    assert info["n_selected_per_class"] == {0: 2, 1: 2, 2: 2}


def test_select_dmrs_thresholds():
    X, y = _dmr_fixture()
    X = X[6:].copy(); y = y[6:]
    assert len(select_dmrs(X, y, 3, n_per_class=2, min_delta=0.5, min_observed_frac=0.5)[0]) == 0
    sel, info = select_dmrs(X, y, 3, n_per_class=1, min_delta=0.1, min_observed_frac=0.5)
    assert len(sel) == 3 and info["n_selected_per_class"] == {0: 1, 1: 1, 2: 1}
    X[:4, 0] = np.nan  # region 0 observed in 2/6 samples -> dropped by min_observed_frac
    sel, info = select_dmrs(X, y, 3, n_per_class=2, min_delta=0.1, min_observed_frac=0.5)
    assert 0 not in sel and info["n_regions_observed_enough"] == 29


DMR_CFG = {"n_per_class": 2, "min_delta": 0.1, "min_observed_frac": 0.5, "max_gap": 100, "C": 1.0}


def test_dmr_logistic():
    X, y = _dmr_fixture()
    noise, signal, y_noise, y_signal = X[:6], X[6:].copy(), y[:6], y[6:]
    signal[0, 20] = np.nan
    clf = DMRClassifier("dmr_logistic", 3, DMR_CFG, seed=0)
    info = clf.fit(signal, y_signal, np.full(30, 3))
    assert info["n_dmrs"] == 6 and info["n_cpgs_in_dmrs"] == 18 and info["classifier"] == "dmr_logistic"
    assert info["thresholds"] == {k: DMR_CFG[k] for k in ("n_per_class", "min_delta", "min_observed_frac", "max_gap")}
    test = X.copy(); test[1, 3] = np.nan  # NaN in a selected region -> training-fold median imputation
    p = clf.predict_proba(test)
    assert p.shape == (12, 3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="no DMR"):
        DMRClassifier("dmr_logistic", 3, DMR_CFG, seed=0).fit(noise, y_noise, np.full(30, 3))


def test_dmr_xgboost():
    pytest.importorskip("xgboost")
    X, y = _dmr_fixture()
    X[7, 20] = np.nan
    clf = DMRClassifier("dmr_xgboost", 3, {**DMR_CFG, "n_estimators": 5, "max_depth": 2}, seed=0)
    info = clf.fit(X, y, np.ones(30, int))
    assert info["imputation"].startswith("none")
    p = clf.predict_proba(X)
    assert p.shape == (12, 3)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)


def test_summary_classifier():
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 2] * 5)
    F = pd.DataFrame({"n_observed_loci": rng.integers(100, 200, 15) + 50 * y, "mean_beta": rng.random(15),
                      "missing_fraction": rng.random(15)})
    clf = SummaryClassifier(seed=0)
    info = clf.fit(F, y)
    assert info["features"] == list(F.columns)
    p = clf.predict_proba(F)
    assert p.shape == (15, 3)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6)
