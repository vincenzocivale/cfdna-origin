"""In-memory read store (per-sample npz files) with label maps, class-balanced training sampling, batch assembly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

MAX_LEN = 32


class ReadStore:
    def __init__(self, reads_dir: Path, samples: pd.DataFrame, device: str):
        self.samples = samples.reset_index(drop=True)
        rows, states, offs, starts = [], [], [], [0]
        base = 0
        for gsm in self.samples.gsm:
            z = np.load(reads_dir / f"{gsm}.npz")
            o = z["read_offsets"]
            rows.append(z["cpg_row"]); states.append(z["state"]); offs.append(o[:-1] + base)
            base += int(o[-1]); starts.append(starts[-1] + len(o) - 1)
        offs.append(np.asarray([base]))
        self.cpg_row = torch.from_numpy(np.concatenate(rows)).to(device)
        self.state = torch.from_numpy(np.concatenate(states)).to(device)
        self.offsets = torch.from_numpy(np.concatenate(offs)).to(device)
        self.sample_start = np.asarray(starts)  # read index range per sample
        self.n_reads = int(self.sample_start[-1])
        self.device = device

    def sample_of_read(self, idx: torch.Tensor) -> torch.Tensor:
        b = torch.from_numpy(self.sample_start[1:]).to(idx.device)
        return torch.bucketize(idx, b, right=True)

    def batch(self, idx: torch.Tensor):
        s = self.offsets[idx]; n = self.offsets[idx + 1] - s
        L = int(min(n.max().item(), MAX_LEN))
        ar = torch.arange(L, device=idx.device)[None]
        mask = ar < n[:, None]
        pos = (s[:, None] + ar).clamp(max=len(self.cpg_row) - 1)
        rows = torch.where(mask, self.cpg_row[pos].long(), torch.zeros_like(pos))
        state = torch.where(mask, self.state[pos].long(), torch.zeros_like(pos))
        return rows, state, mask

    def balanced_indices(self, organ_of_sample: np.ndarray, batch_size: int, gen: torch.Generator) -> torch.Tensor:
        """Draw organ uniformly -> sample uniformly within organ -> read uniformly within sample."""
        organs = np.unique(organ_of_sample)
        per = [np.flatnonzero(organ_of_sample == o) for o in organs]
        o = torch.randint(len(organs), (batch_size,), generator=gen)
        picks = torch.empty(batch_size, dtype=torch.long)
        for j, lst in enumerate(per):
            m = (o == j).nonzero().squeeze(1)
            if len(m):
                picks[m] = torch.from_numpy(lst)[torch.randint(len(lst), (len(m),), generator=gen)]
        st, en = self.sample_start[picks.numpy()], self.sample_start[picks.numpy() + 1]
        u = torch.rand(batch_size, generator=gen).numpy()
        return torch.from_numpy((st + (u * (en - st)).astype(np.int64))).to(self.device)

    def eval_indices(self, sample_i: int, max_reads: int, seed: int = 0) -> torch.Tensor:
        st, en = int(self.sample_start[sample_i]), int(self.sample_start[sample_i + 1])
        g = np.random.default_rng(seed + sample_i)
        idx = np.arange(st, en) if en - st <= max_reads else np.sort(g.choice(np.arange(st, en), max_reads, replace=False))
        return torch.from_numpy(idx).to(self.device)
