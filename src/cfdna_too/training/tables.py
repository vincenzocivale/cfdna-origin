"""Frozen locus tables for each experimental arm; only the table changes between arms."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from cfdna_too.data.cpg_index import CpGIndex


def load_table(arm: str, h5_path: Path, index: CpGIndex, device: str, seed: int = 0):
    """Returns (table [N,D] fp16 on device or a dummy for 'none', positions [N] int32 on device, use_embedding)."""
    n = index.offsets[-2]  # universe: chr1-22,X,Y (chrM is the last block and is excluded)
    positions = torch.from_numpy(index.positions[:n].astype(np.int32)).to(device)
    if arm == "none":
        return torch.zeros(1, 1, dtype=torch.float16, device=device), positions, False
    if arm == "random":
        g = torch.Generator(device=device).manual_seed(seed)
        t = torch.empty(n, 256, dtype=torch.float16, device=device)
        for s in range(0, n, 2_000_000):
            t[s : s + 2_000_000] = torch.randn(min(2_000_000, n - s), 256, device=device, generator=g).half()
        return t, positions, True
    if arm == "functional":
        with h5py.File(h5_path, "r") as f:
            if f["embedding"].shape[0] != n:
                raise ValueError(f"embedding rows {f['embedding'].shape[0]} != universe {n}")
            keys = f["cpg_idx"][:]
            if not np.array_equal(keys & 0xFFFFFFFF, index.positions[:n]):
                raise ValueError("embedding row order differs from the CpG index")
            emb = f["embedding"]
            sample = emb[np.sort(np.random.default_rng(0).choice(n, 200_000, replace=False))].astype(np.float32)
            scale = torch.from_numpy(sample.std(0) + 1e-6).to(device)
            t = torch.empty(n, emb.shape[1], dtype=torch.float16, device=device)
            for s in range(0, n, 1_000_000):
                t[s : s + 1_000_000] = (torch.from_numpy(emb[s : s + 1_000_000]).to(device).float() / scale).half()
        return t, positions, True
    raise ValueError(arm)
