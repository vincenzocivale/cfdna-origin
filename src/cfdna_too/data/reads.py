"""PAT -> sampled read table.

PAT line: chrom, 1-based global CpG index of the first CpG, pattern over consecutive CpGs (C=methylated,
T=unmethylated, .=CpG not covered), count. A read keeps only its called CpGs. Reads are thinned by Binomial(count, p)
(weight = surviving copies), reads with fewer than `min_cpg` or more than `max_cpg` called CpGs are dropped, and
chrM / non-universe lines are counted and dropped (the embedding universe is chr1-22,X,Y).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .cpg_index import CHROMS, CpGIndex

UNIVERSE_CHROMS = [c for c in CHROMS if c != "chrM"]


def extract_sample(pat: Path, index: CpGIndex, *, p: float, min_cpg: int, max_cpg: int, seed_key: str,
                   chunk_rows: int = 5_000_000) -> tuple[dict, dict]:
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    n_universe = int(index.offsets[len(UNIVERSE_CHROMS)])
    stats = dict(lines=0, reads=0, dropped_nonuniverse_lines=0, dropped_short_reads=0, dropped_long_reads=0,
                 dropped_boundary_lines=0, kept_lines=0, kept_reads=0)
    offsets, rows, states, weights = [0], [], [], []
    reader = pd.read_csv(pat, sep="\t", header=None, names=["chrom", "start", "pat", "count"],
                         dtype={"chrom": "category", "start": np.int64, "pat": str, "count": np.int64},
                         chunksize=chunk_rows, compression="gzip")
    for ch in reader:
        stats["lines"] += len(ch); stats["reads"] += int(ch["count"].sum())
        ok = ch["chrom"].isin(UNIVERSE_CHROMS).to_numpy() & (ch["start"].to_numpy() <= n_universe)
        stats["dropped_nonuniverse_lines"] += int((~ok).sum())
        ch = ch[ok]
        pats = ch["pat"]
        called = pats.str.len() - pats.str.count(r"\.")
        short = (called < min_cpg).to_numpy(); long_ = (called > max_cpg).to_numpy()
        stats["dropped_short_reads"] += int(ch["count"].to_numpy()[short].sum())
        stats["dropped_long_reads"] += int(ch["count"].to_numpy()[long_].sum())
        ch = ch[~short & ~long_]
        k = rng.binomial(ch["count"].to_numpy(), p)
        ch, k = ch[k > 0], k[k > 0]
        if not len(ch):
            continue
        start = ch["start"].to_numpy() - 1  # 0-based global row of the first CpG
        seqs = ch["pat"].tolist()
        lens = np.fromiter((len(s) for s in seqs), np.int64, len(seqs))
        # a read must stay inside its chromosome's CpG range
        chrom_row = np.searchsorted(index.offsets, start, side="right") - 1
        end_row = start + lens - 1
        inside = (end_row < index.offsets[chrom_row + 1]) & (np.asarray([CHROMS.index(c) for c in ch["chrom"].astype(str)]) == chrom_row)
        stats["dropped_boundary_lines"] += int((~inside).sum())
        seqs = [s for s, m in zip(seqs, inside) if m]
        start, lens, k = start[inside], lens[inside], k[inside]
        chars = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)
        within = np.arange(len(chars)) - np.repeat(np.cumsum(lens) - lens, lens)
        row_all = np.repeat(start, lens) + within
        called_mask = (chars == ord("C")) | (chars == ord("T"))
        if not np.isin(chars, np.frombuffer(b"CT.", np.uint8)).all():
            raise ValueError(f"{pat.name}: unexpected pattern symbol")
        n_called = np.add.reduceat(called_mask.astype(np.int64), np.cumsum(lens) - lens)
        rows.append(row_all[called_mask].astype(np.int32))
        states.append((chars[called_mask] == ord("C")).astype(np.uint8))
        weights.append(k.astype(np.uint16))
        offsets.extend((offsets[-1] + np.cumsum(n_called)).tolist())
        stats["kept_lines"] += len(seqs); stats["kept_reads"] += int(k.sum())
    out = dict(read_offsets=np.asarray(offsets, np.int64),
               cpg_row=np.concatenate(rows) if rows else np.empty(0, np.int32),
               state=np.concatenate(states) if states else np.empty(0, np.uint8),
               weight=np.concatenate(weights) if weights else np.empty(0, np.uint16))
    return out, stats
