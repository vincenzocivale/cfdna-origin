import pytest
import torch

from cfdna_origin.models.classifier import MILClassifier, count_parameters
from cfdna_origin.models.fragment.encoders import FRAGMENT_ENCODERS, token_dropout
from cfdna_origin.models.sample.aggregators import AGGREGATORS, StreamingPool, segment_softmax_pool

D = 16
AGG_CFGS = {
    "mean": {"type": "mean"},
    "gated_branches": {"type": "gated_attention", "attn_dim": 8, "class_branches": True},
    "gated_single": {"type": "gated_attention", "attn_dim": 8, "class_branches": False},
    "pma": {"type": "pma", "n_heads": 4, "n_seeds": 2},
}
ENC_CFGS = {
    "sab_pma": {"type": "set_attention", "n_layers": 1, "n_heads": 4, "ffn_dim": 32, "dropout": 0.1, "readout": "pma"},
    "sab_mean": {"type": "set_attention", "n_layers": 1, "n_heads": 4, "ffn_dim": 32, "dropout": 0.1,
                 "readout": "mean"},
    "deepsets": {"type": "deepsets", "hidden": 32, "dropout": 0.1},
}


def _cfg(enc="sab_pma", agg="gated_branches", **token):
    return {"d_model": D, "token": {"dropout": 0.1, "keep_min": 3, **token}, "fragment_encoder": ENC_CFGS[enc],
            "aggregator": AGG_CFGS[agg], "head_dropout": 0.25}


def _batch(F=10, L=6, locus_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = torch.randint(1, L + 1, (F,), generator=g)
    mask = torch.arange(L)[None] < n[:, None]
    emb = torch.randn(F, L, locus_dim, generator=g) if locus_dim else None
    state = torch.randint(0, 2, (F, L), generator=g)
    geom = torch.rand(F, L, 3, generator=g)
    return emb, state, geom, mask


def test_token_dropout_keeps_min_and_never_adds():
    torch.manual_seed(0)
    mask = torch.arange(10)[None] < torch.tensor([1, 2, 3, 5, 10] * 40)[:, None]
    for p in (0.5, 0.9, 1.0):
        out = token_dropout(mask, p, keep_min=3)
        assert not (out & ~mask).any()
        assert torch.all(out.sum(1) >= torch.clamp(mask.sum(1), max=3))
    assert torch.equal(token_dropout(mask, 0.0, 3), mask)
    assert (token_dropout(mask, 1.0, 3).sum(1) == torch.clamp(mask.sum(1), max=3)).all()


@pytest.mark.parametrize("enc", list(ENC_CFGS))
def test_fragment_encoder_shape_and_padding_invariance(enc):
    torch.manual_seed(0)
    cfg = dict(ENC_CFGS[enc]); model = FRAGMENT_ENCODERS[cfg.pop("type")](d_model=D, **cfg).eval()
    tokens = torch.randn(7, 5, D)
    mask = torch.arange(5)[None] < torch.tensor([1, 2, 3, 4, 5, 5, 3])[:, None]
    out = model(tokens, mask)
    assert out.shape == (7, model.out_dim)
    padded = torch.cat([tokens, torch.randn(7, 4, D) * 100], 1)  # extra padded positions with junk content
    out2 = model(padded, torch.cat([mask, torch.zeros(7, 4, dtype=torch.bool)], 1))
    torch.testing.assert_close(out, out2, atol=1e-5, rtol=1e-5)


def _pool(name, n_classes=3):
    cfg = dict(AGG_CFGS[name])
    return AGGREGATORS[cfg.pop("type")](D, n_classes=n_classes, **cfg).eval()


def test_segment_softmax_weights_sum_to_one():
    g = torch.Generator().manual_seed(0)
    scores, values = torch.randn(9, 2, generator=g) * 5, torch.randn(9, 2, 4, generator=g)
    seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pooled, w = segment_softmax_pool(scores, values, seg, 3)
    sums = torch.zeros(3, 2).index_add_(0, seg, w)
    torch.testing.assert_close(sums, torch.ones(3, 2))
    torch.testing.assert_close(pooled[1], (torch.softmax(scores[3:5], 0)[..., None] * values[3:5]).sum(0))


@pytest.mark.parametrize("name", list(AGG_CFGS))
def test_aggregator_shapes_and_permutation_invariance(name):
    torch.manual_seed(0)
    pool = _pool(name)
    x = torch.randn(12, D)
    seg = torch.tensor([0] * 5 + [1] * 7)
    z, w = pool(x, seg, 2)
    expected = {"mean": (2, D), "gated_branches": (2, 3, D), "gated_single": (2, D), "pma": (2, 2 * D)}[name]
    assert z.shape == expected and w.shape == (12, pool.n_heads)
    torch.testing.assert_close(torch.zeros(2, pool.n_heads).index_add_(0, seg, w), torch.ones(2, pool.n_heads))
    perm = torch.cat([torch.randperm(5), 5 + torch.randperm(7)])
    z2, _ = pool(x[perm], seg, 2)
    torch.testing.assert_close(z, z2, atol=1e-5, rtol=1e-5)
    interleaved = torch.randperm(12)  # fragments of different samples interleaved
    z3, _ = pool(x[interleaved], seg[interleaved], 2)
    torch.testing.assert_close(z, z3, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", list(AGG_CFGS))
def test_streaming_pool_equals_one_shot(name):
    torch.manual_seed(0)
    pool = _pool(name)
    x = torch.randn(23, D) * 3
    one_shot, _ = pool(x, torch.zeros(23, dtype=torch.long), 1)
    acc = StreamingPool(pool)
    for chunk in (x[:4], x[4:5], x[5:20], x[20:]):
        acc.update(chunk)
    torch.testing.assert_close(acc.finalize(), one_shot[0], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        StreamingPool(pool).finalize()


@pytest.mark.parametrize("agg", list(AGG_CFGS))
@pytest.mark.parametrize("enc", list(ENC_CFGS))
def test_classifier_predict_streaming_equals_forward(enc, agg):
    torch.manual_seed(0)
    model = MILClassifier(8, 3, _cfg(enc, agg)).eval()
    emb, st, ge, ma = _batch(F=11)
    logits, w = model(emb, st, ge, ma, torch.zeros(11, dtype=torch.long), 1)
    assert logits.shape == (1, 3)
    chunks = [(emb[s:s + 4], st[s:s + 4], ge[s:s + 4], ma[s:s + 4]) for s in range(0, 11, 4)]
    torch.testing.assert_close(model.predict_streaming(chunks), logits[0], atol=1e-5, rtol=1e-5)


def test_classifier_training_forward_backward_multi_sample():
    torch.manual_seed(0)
    model = MILClassifier(8, 3, _cfg()).train()
    emb, st, ge, ma = _batch(F=12)
    logits, _ = model(emb, st, ge, ma, torch.tensor([0] * 6 + [1] * 6), 2)
    assert logits.shape == (2, 3)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.tokens.locus.parameters())


def test_methylation_only_model():
    torch.manual_seed(0)
    model = MILClassifier(0, 3, _cfg()).eval()
    _, st, ge, ma = _batch(F=5, locus_dim=0)
    logits, _ = model(None, st, ge, ma, torch.zeros(5, dtype=torch.long), 1)
    assert logits.shape == (1, 3) and torch.isfinite(logits).all()
    assert count_parameters(model)["representation_adapter"] == D


def test_frozen_table_not_a_parameter_and_equal_adapters():
    a, b = MILClassifier(8, 3, _cfg()), MILClassifier(8, 3, _cfg())
    ca, cb = count_parameters(a), count_parameters(b)
    assert ca == cb and ca["representation_adapter"] == 8 * D + D + 2 * D  # Linear(8->D) + LayerNorm
    assert ca["trainable"] == sum(p.numel() for p in a.parameters())
    fixed = MILClassifier(300, 3, _cfg(fixed_projection_dim=8))  # fixed random projection is a buffer, not trained
    assert count_parameters(fixed) == ca
    assert "tokens.locus.fixed_proj" in dict(fixed.named_buffers())


def test_use_methylation_state_false_ignores_states():
    torch.manual_seed(0)
    model = MILClassifier(8, 3, _cfg(use_methylation_state=False)).eval()
    emb, st, ge, ma = _batch(F=6)
    seg = torch.zeros(6, dtype=torch.long)
    torch.testing.assert_close(model(emb, st, ge, ma, seg, 1)[0], model(emb, 1 - st, ge, ma, seg, 1)[0])
    model_state = MILClassifier(8, 3, _cfg()).eval()
    assert not torch.allclose(model_state(emb, st, ge, ma, seg, 1)[0], model_state(emb, 1 - st, ge, ma, seg, 1)[0])
