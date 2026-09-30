"""Level 1 — fragment encoders.

Each CpG token = adapter(frozen locus embedding) + state embedding (0/1) + linear(geometry). The adapter is the only
representation-dependent module:
- dim > 0: Linear(dim -> d_model) -> LayerNorm. Representations are materialised as a label-free PCA to a common
  width r (see `representations/registry.py`), so trainable parameters are identical for every arm with a locus
  embedding. (An optional fixed, non-trainable random projection is available for artifacts used without PCA.)
- dim == 0 (`methylation_only`): a single learned constant vector (no locus information).
The frozen embedding itself is never a parameter of the model.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LocusAdapter(nn.Module):
    def __init__(self, in_dim: int, d_model: int, *, fixed_projection_dim: int | None = None, seed: int = 0):
        super().__init__()
        self.in_dim = in_dim
        if in_dim == 0:
            self.constant = nn.Parameter(torch.zeros(d_model))
            return
        if fixed_projection_dim:
            g = torch.Generator().manual_seed(seed)
            proj = torch.randn(in_dim, fixed_projection_dim, generator=g) / math.sqrt(in_dim)
            self.register_buffer("fixed_proj", proj)  # frozen, identical for any arm of equal in_dim and seed
            width = fixed_projection_dim
        else:
            self.fixed_proj = None
            width = in_dim
        self.net = nn.Sequential(nn.Linear(width, d_model), nn.LayerNorm(d_model))

    def forward(self, emb: torch.Tensor | None, shape: tuple[int, int]) -> torch.Tensor:
        if self.in_dim == 0:
            return self.constant.expand(*shape, -1)
        x = emb.float()
        if self.fixed_proj is not None:
            x = x @ self.fixed_proj
        return self.net(x)


class TokenEmbedding(nn.Module):
    def __init__(self, locus_dim: int, d_model: int, *, fixed_projection_dim: int | None = None, seed: int = 0):
        super().__init__()
        self.locus = LocusAdapter(locus_dim, d_model, fixed_projection_dim=fixed_projection_dim, seed=seed)
        self.state = nn.Embedding(2, d_model)
        self.geom = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, emb, state, geom):
        return self.locus(emb, state.shape) + self.state(state) + self.geom(geom)


def token_dropout(mask: torch.Tensor, p: float, keep_min: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly drop CpG tokens (training only) while keeping at least min(keep_min, n) tokens per fragment."""
    if p <= 0:
        return mask
    u = torch.rand(mask.shape, device=mask.device, generator=generator)
    drop = (u < p) & mask
    n = mask.sum(1, keepdim=True)
    # rank tokens by u among real tokens; the keep_min largest-u real tokens are never dropped
    order = torch.where(mask, u, torch.full_like(u, -1.0)).argsort(1, descending=True).argsort(1)
    protected = order < torch.minimum(n, torch.full_like(n, keep_min))
    return mask & ~(drop & ~protected)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)


class DeepSetsFragmentEncoder(nn.Module):
    """rho([mean_i phi(t_i), max_i phi(t_i)]) (Zaheer et al. 2017). Order enters only through geometry features."""

    def __init__(self, d_model: int = 64, hidden: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.rho = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU())
        self.out_dim = d_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(tokens)
        return self.rho(torch.cat([_masked_mean(h, mask), _masked_max(h, mask)], -1))


class SetAttentionFragmentEncoder(nn.Module):
    """Set Transformer style (Lee et al. 2019): `n_layers` pre-norm SAB blocks over the CpG tokens, then a readout:
    PMA with one learned seed (default) or masked mean."""

    def __init__(self, d_model: int = 64, n_layers: int = 1, n_heads: int = 4, ffn_dim: int = 128,
                 dropout: float = 0.1, readout: str = "pma", **_):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.readout = readout
        if readout == "pma":
            self.seed = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
            self.pma = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.pma_ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(),
                                        nn.Linear(ffn_dim, d_model))
        elif readout != "mean":
            raise ValueError(f"unknown readout {readout!r}")
        self.out_dim = d_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.encoder(tokens, src_key_padding_mask=~mask))
        if self.readout == "mean":
            return _masked_mean(h, mask)
        z, _ = self.pma(self.seed.expand(len(h), -1, -1), h, h, key_padding_mask=~mask, need_weights=False)
        z = z[:, 0] + self.seed[0]
        return z + self.pma_ff(z)


FRAGMENT_ENCODERS = {"set_attention": SetAttentionFragmentEncoder, "deepsets": DeepSetsFragmentEncoder}
