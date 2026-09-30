"""Hierarchical MIL classifier: CpG tokens -> fragment encoder -> sample aggregator -> class logits.

The frozen locus table is NOT part of the module: the caller gathers the embedding rows of a bag (`LocusTable`) and
passes them in, so the model's parameters never include (and can never update) the representation.
"""
from __future__ import annotations

import torch
from torch import nn

from cfdna_origin.models.fragment.encoders import FRAGMENT_ENCODERS, TokenEmbedding, token_dropout
from cfdna_origin.models.sample.aggregators import AGGREGATORS, StreamingPool


class ClassWiseHead(nn.Module):
    """logit_c = <w_c, dropout(LN(z_c))> + b_c for class-branch pooled embeddings z [B, C, d]."""

    def __init__(self, d: int, n_classes: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dropout)
        self.w = nn.Parameter(torch.randn(n_classes, d) * d ** -0.5); self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, z):
        return (self.drop(self.norm(z)) * self.w).sum(-1) + self.b


class MILClassifier(nn.Module):
    def __init__(self, locus_dim: int, n_classes: int, cfg: dict):
        super().__init__()
        d = cfg.get("d_model", 64)
        tok = cfg.get("token", {})
        self.use_state = tok.get("use_methylation_state", True)
        self.token_dropout, self.keep_min = tok.get("dropout", 0.0), tok.get("keep_min", 3)
        self.tokens = TokenEmbedding(locus_dim, d, fixed_projection_dim=tok.get("fixed_projection_dim"),
                                     seed=tok.get("projection_seed", 0))
        fe = dict(cfg.get("fragment_encoder", {"type": "set_attention"}))
        self.fragment = FRAGMENT_ENCODERS[fe.pop("type")](d_model=d, **fe)
        ag = dict(cfg.get("aggregator", {"type": "gated_attention"}))
        self.pool = AGGREGATORS[ag.pop("type")](self.fragment.out_dim, n_classes=n_classes, **ag)
        drop = cfg.get("head_dropout", 0.25)
        if self.pool.class_branches:
            self.head = ClassWiseHead(self.fragment.out_dim, n_classes, drop)
        else:
            self.head = nn.Sequential(nn.LayerNorm(self.pool.out_dim), nn.Dropout(drop), nn.Linear(self.pool.out_dim, n_classes))

    def encode_fragments(self, emb, state, geom, mask):
        if not self.use_state:  # "coverage only" ablation: which loci are observed, not their methylation
            state = torch.zeros_like(state)
        if self.training:
            mask = token_dropout(mask, self.token_dropout, self.keep_min)
        return self.fragment(self.tokens(emb, state, geom), mask)

    def forward(self, emb, state, geom, mask, seg, n_samples):
        h = self.encode_fragments(emb, state, geom, mask)
        z, w = self.pool(h, seg, n_samples)
        return self.head(z), w

    @torch.no_grad()
    def predict_streaming(self, chunks) -> torch.Tensor:
        """Exact aggregation of one sample from an iterable of (emb, state, geom, mask) fragment chunks."""
        acc = StreamingPool(self.pool)
        for emb, state, geom, mask in chunks:
            acc.update(self.encode_fragments(emb, state, geom, mask))
        return self.head(acc.finalize()[None])[0]


def count_parameters(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(p.numel() for p in model.tokens.locus.parameters() if p.requires_grad)
    return {"trainable": int(trainable), "representation_adapter": int(adapter), "shared": int(trainable - adapter)}
