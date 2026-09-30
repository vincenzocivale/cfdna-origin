"""Level 2 — sample (MIL) aggregators over fragment embeddings.

Every aggregator is written as H-head softmax-attention pooling: per fragment a score s_h(x) and a value v_h(x);
pooled_h = sum_i softmax_i(s_h) v_h(x_i); then `finalize` maps the pooled heads to the sample embedding.
- `mean`:            s = 0, v = x                         (mean pooling baseline)
- `gated_attention`: s_c = w_c^T(tanh(Vx) * sigmoid(Ux)), v = x (Ilse et al. 2018); with `class_branches` one
                     attention branch per class and a class-wise linear head (CLAM, Lu et al. 2021)
- `pma`:             learned seed queries, multi-head dot-product attention + rFF (Set Transformer PMA, Lee et al.
                     2019); with one seed this is learned-query pooling.
Because softmax pooling decomposes with a running log-sum-exp, a sample can be aggregated exactly over arbitrarily
many fragments in chunks (`StreamingPool`), so evaluation never needs all fragments in GPU memory.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def segment_softmax_pool(scores: torch.Tensor, values: torch.Tensor, seg: torch.Tensor, n_seg: int):
    """scores [N,H], values [N,H,V], seg [N] -> pooled [n_seg,H,V], weights [N,H] (softmax within each segment)."""
    H = scores.shape[1]
    mx = torch.full((n_seg, H), -torch.inf, device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, seg[:, None].expand(-1, H), scores, reduce="amax", include_self=True)
    e = torch.exp(scores - mx[seg])
    den = torch.zeros((n_seg, H), device=scores.device, dtype=scores.dtype).index_add_(0, seg, e)
    w = e / den[seg].clamp(min=1e-30)
    pooled = torch.zeros((n_seg, H, values.shape[2]), device=values.device, dtype=values.dtype)
    pooled.index_add_(0, seg, w.unsqueeze(-1) * values)
    return pooled, w


class _AttentionPool(nn.Module):
    n_heads: int = 1
    class_branches: bool = False  # True: finalize returns [B, n_classes, d] consumed by a class-wise head
    instance_dropout: float = 0.0

    def scores_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def finalize(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled.flatten(1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor, n_seg: int):
        s, v = self.scores_values(x)
        s = s.float()
        if self.training and self.instance_dropout > 0:  # drop whole fragments from the softmax
            drop = torch.rand(len(s), device=s.device) < self.instance_dropout
            s = s.masked_fill(drop[:, None], -1e9)
        pooled, w = segment_softmax_pool(s, v.float(), seg, n_seg)
        return self.finalize(pooled), w


class MeanPool(_AttentionPool):
    def __init__(self, d_in: int, instance_dropout: float = 0.0, **_):
        super().__init__()
        self.instance_dropout = instance_dropout
        self.out_dim = d_in

    def scores_values(self, x):
        return torch.zeros(len(x), 1, device=x.device, dtype=x.dtype), x.unsqueeze(1)


class GatedAttentionPool(_AttentionPool):
    def __init__(self, d_in: int, attn_dim: int = 32, n_classes: int = 1, class_branches: bool = True,
                 dropout: float = 0.0, instance_dropout: float = 0.0, **_):
        super().__init__()
        self.class_branches = bool(class_branches)
        self.n_heads = n_classes if class_branches else 1
        self.instance_dropout = instance_dropout
        self.V = nn.Linear(d_in, attn_dim); self.U = nn.Linear(d_in, attn_dim); self.w = nn.Linear(attn_dim, self.n_heads)
        self.drop = nn.Dropout(dropout)
        self.out_dim = d_in

    def scores_values(self, x):
        s = self.w(self.drop(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))))
        return s, x.unsqueeze(1).expand(-1, self.n_heads, -1)

    def finalize(self, pooled):
        return pooled if self.class_branches else pooled.flatten(1)


class PMAPool(_AttentionPool):
    def __init__(self, d_in: int, n_heads: int = 4, n_seeds: int = 1, dropout: float = 0.0,
                 instance_dropout: float = 0.0, **_):
        super().__init__()
        self.instance_dropout = instance_dropout
        if d_in % n_heads:
            raise ValueError("d_in must be divisible by n_heads")
        self.n_seeds, self.heads, self.dh = n_seeds, n_heads, d_in // n_heads
        self.n_heads = n_seeds * n_heads
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_in) / math.sqrt(d_in))
        self.k = nn.Linear(d_in, d_in); self.v = nn.Linear(d_in, d_in); self.o = nn.Linear(d_in, d_in)
        self.norm1 = nn.LayerNorm(d_in); self.norm2 = nn.LayerNorm(d_in)
        self.ff = nn.Sequential(nn.Linear(d_in, 2 * d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_in, d_in))
        self.out_dim = n_seeds * d_in

    def scores_values(self, x):
        k = self.k(x).view(len(x), self.heads, self.dh)
        q = self.seeds.view(self.n_seeds, self.heads, self.dh)
        s = torch.einsum("nhd,shd->nsh", k, q) / math.sqrt(self.dh)  # [N, seeds, heads]
        v = self.v(x).view(len(x), 1, self.heads, self.dh).expand(-1, self.n_seeds, -1, -1)
        return s.reshape(len(x), -1), v.reshape(len(x), self.n_heads, self.dh)

    def finalize(self, pooled):
        B = len(pooled)
        att = pooled.view(B, self.n_seeds, self.heads * self.dh)
        h = self.norm1(self.seeds[None] + self.o(att))
        h = self.norm2(h + self.ff(h))
        return h.flatten(1)


AGGREGATORS = {"mean": MeanPool, "gated_attention": GatedAttentionPool, "pma": PMAPool}


class StreamingPool:
    """Exact chunked evaluation of an `_AttentionPool` for ONE sample (running log-sum-exp)."""

    def __init__(self, pool: _AttentionPool):
        self.pool = pool
        self.m = self.s = self.v = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        sc, val = self.pool.scores_values(x)
        sc, val = sc.float(), val.float()
        cm = sc.max(0).values  # [H]
        m = cm if self.m is None else torch.maximum(self.m, cm)
        e = torch.exp(sc - m)
        s_new = e.sum(0)
        v_new = (e.unsqueeze(-1) * val).sum(0)
        if self.m is not None:
            r = torch.exp(self.m - m)
            s_new = s_new + self.s * r
            v_new = v_new + self.v * r.unsqueeze(-1)
        self.m, self.s, self.v = m, s_new, v_new

    @torch.no_grad()
    def finalize(self) -> torch.Tensor:
        if self.m is None:
            raise ValueError("no fragments were aggregated")
        return self.pool.finalize((self.v / self.s.unsqueeze(-1))[None])[0]
