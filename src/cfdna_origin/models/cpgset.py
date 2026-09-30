"""CpG-set -> sample classifier for the processed-beta pilot (no fragments).

A sample is the set of its observed CpGs; token_i = [adapter(frozen locus embedding_i) | value(beta_i, coverage_i)].
Architectures (all permutation invariant, small; exact chunked evaluation over all observed CpGs):
  mean      one-layer token map (GELU(Linear)) -> mean pooling -> linear head
  deepsets  2-layer phi -> mean pooling -> rho MLP -> linear head          (Zaheer et al. 2017)
  pma       2-layer phi -> PMA (1 learned query, 4 heads) -> linear head   (Lee et al. 2019)
The locus adapter is the only representation-dependent module (identical size across arms after the common PCA);
`methylation_only` uses a learned constant locus vector. The frozen locus table is never a model parameter.
"""
from __future__ import annotations

import torch
from torch import nn

from cfdna_origin.models.fragment.encoders import LocusAdapter
from cfdna_origin.models.sample.aggregators import MeanPool, PMAPool, StreamingPool


class CpGSetClassifier(nn.Module):
    def __init__(self, locus_dim: int, n_classes: int, cfg: dict):
        super().__init__()
        d, h = cfg.get("d_model", 32), cfg.get("hidden", 64)
        drop = cfg.get("dropout", 0.1)
        self.arch = cfg["arch"]
        self.use_coverage = cfg.get("use_coverage", True)
        self.locus = LocusAdapter(locus_dim, d)
        self.value = nn.Linear(2 if self.use_coverage else 1, d)
        if self.arch == "mean":
            self.phi = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
            self.pool = MeanPool(d)
            self.rho = nn.Identity()
        elif self.arch == "deepsets":
            self.phi = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, d))
            self.pool = MeanPool(d)
            self.rho = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        elif self.arch == "pma":
            self.phi = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, d))
            self.pool = PMAPool(d, n_heads=cfg.get("n_heads", 4), n_seeds=1)
            self.rho = nn.Identity()
        else:
            raise ValueError(f"unknown arch {self.arch!r}")
        self.head = nn.Sequential(nn.LayerNorm(self.pool.out_dim), nn.Dropout(cfg.get("head_dropout", 0.25)),
                                  nn.Linear(self.pool.out_dim, n_classes))

    def encode(self, emb: torch.Tensor | None, beta: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Per-token representations [N, d]; beta in [0,1], cov = read depth (>0)."""
        v = beta[:, None] - 0.5
        if self.use_coverage:
            v = torch.cat([v, torch.log1p(cov.float())[:, None] / 5.0], 1)
        x = torch.cat([self.locus(emb, (len(beta),)), self.value(v)], 1)
        return self.phi(x)

    def classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(self.rho(pooled))

    def forward(self, emb, beta, cov, seg, n_samples):
        z, w = self.pool(self.encode(emb, beta, cov), seg, n_samples)
        return self.classify(z), w

    @torch.no_grad()
    def predict_streaming(self, chunks) -> torch.Tensor:
        acc = StreamingPool(self.pool)
        for emb, beta, cov in chunks:
            acc.update(self.encode(emb, beta, cov))
        return self.classify(acc.finalize()[None])[0]

    @torch.no_grad()
    def token_scores(self, emb, beta, cov) -> tuple[torch.Tensor, torch.Tensor]:
        """Contribution proxy per token: (pooling score, class logits of the token alone). Interpretation only."""
        x = self.encode(emb, beta, cov)
        s, v = self.pool.scores_values(x)
        solo = self.classify(self.pool.finalize(v.float()))
        return s.float().mean(1), solo
