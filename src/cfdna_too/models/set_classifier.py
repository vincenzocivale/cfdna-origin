"""Read-level set classifier over frozen locus embeddings.

Each read is a set of called CpGs; a token = [projected frozen locus embedding | methylation state | genomic
geometry (distance from first CpG, gap to previous CpG, relative rank)]. A small transformer with a CLS token
aggregates the set. Two heads: organ (coarse) and cell type (fine). Only this classifier is trained; the locus
embedding table is a frozen buffer that the model never updates.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SetClassifier(nn.Module):
    def __init__(self, table: torch.Tensor, positions: torch.Tensor, n_organ: int, n_fine: int, *,
                 d_model: int = 192, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1, use_embedding: bool = True):
        super().__init__()
        self.register_buffer("table", table, persistent=False)  # [N, D] frozen, fp16 on device
        self.register_buffer("positions", positions, persistent=False)  # [N] int32 genomic position
        self.use_embedding = use_embedding
        self.proj = nn.Sequential(nn.LayerNorm(table.shape[1]), nn.Linear(table.shape[1], d_model)) if use_embedding else None
        self.state = nn.Embedding(2, d_model)
        self.geom = nn.Linear(3, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.organ_head = nn.Linear(d_model, n_organ)
        self.fine_head = nn.Linear(d_model, n_fine)

    def forward(self, rows: torch.Tensor, state: torch.Tensor, mask: torch.Tensor):
        """rows [B,L] long, state [B,L] long {0,1}, mask [B,L] bool (True = real CpG)."""
        pos = self.positions[rows].float()
        first = pos[:, :1]
        prev = torch.cat([pos[:, :1], pos[:, :-1]], 1)
        rank = torch.arange(rows.shape[1], device=rows.device)[None].float() / max(rows.shape[1] - 1, 1)
        geom = torch.stack([torch.log1p((pos - first).clamp(min=0)) / 7.0,
                            torch.log1p((pos - prev).clamp(min=0)) / 7.0, rank.expand_as(pos)], -1)
        x = self.state(state) + self.geom(geom)
        if self.use_embedding:
            x = x + self.proj(self.table[rows].float())
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
        return self.organ_head(h), self.fine_head(h)
