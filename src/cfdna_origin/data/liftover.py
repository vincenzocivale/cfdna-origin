"""Vectorised point liftover of CpG locus keys between assemblies with a UCSC chain file (e.g. hg19ToHg38).

Each ungapped chain block maps [t, t+size) on the source to the target. A point covered by more than one block
(overlapping chains) is ambiguous and dropped. On a minus-strand block the CpG maps to the opposite strand, so its
C lands on the G of the target CpG: the target C position is (mapped position - 1). Finally a lifted locus is kept only
if it is a CpG in the target assembly (`target_cpg_keys`). Every drop is counted; nothing is dropped silently.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from cfdna_origin.data.loci import CHROM_CODE, key_chrom_code, key_position, normalize_chrom


class Chain:
    def __init__(self, path: Path):
        blocks = []  # (src_code, src_start0, size, dst_code, dst_start0_fwd, strand(+1/-1), dst_size)
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            header = None
            for line in fh:
                f = line.split()
                if not f:
                    header = None; continue
                if f[0] == "chain":
                    t_chr, q_chr = normalize_chrom(f[2]), normalize_chrom(f[7])
                    ok = t_chr in CHROM_CODE and q_chr in CHROM_CODE
                    header = dict(t=CHROM_CODE.get(t_chr), q=CHROM_CODE.get(q_chr), qsize=int(f[8]), qstrand=f[9],
                                  tpos=int(f[5]), qpos=int(f[10])) if ok else None
                    continue
                if header is None:
                    continue
                size = int(f[0])
                qfwd = header["qpos"] if header["qstrand"] == "+" else header["qsize"] - header["qpos"] - size
                blocks.append((header["t"], header["tpos"], size, header["q"], qfwd, 1 if header["qstrand"] == "+" else -1))
                if len(f) == 3:
                    header["tpos"] += size + int(f[1]); header["qpos"] += size + int(f[2])
        b = np.asarray(blocks, dtype=np.int64)
        order = np.lexsort((b[:, 1], b[:, 0]))
        self.b = b[order]
        self.gstart = (self.b[:, 0] << 32) | self.b[:, 1]  # global sortable start
        self.gend = self.gstart + self.b[:, 2]
        self.prev_max_end = np.maximum.accumulate(np.concatenate([[-1], self.gend[:-1]]))
        self.max_size = int(self.b[:, 2].max())

    def lift(self, keys: np.ndarray, target_cpg_keys: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """keys (source, 1-based C) -> (target keys, kept mask over input, stats)."""
        keys = np.asarray(keys, np.int64)
        p0 = (key_chrom_code(keys) << 32) | (key_position(keys) - 1)  # 0-based global
        i = np.searchsorted(self.gstart, p0, side="right") - 1
        valid = i >= 0
        ic = np.clip(i, 0, None)
        inside = valid & (p0 < self.gend[ic])
        earlier = valid & (self.prev_max_end[ic] > p0)  # some earlier-starting block also covers the point
        ambiguous = inside & earlier
        for k in np.flatnonzero(~inside & earlier):  # covered only by earlier (overlapping/nested) blocks: rare
            lo = np.searchsorted(self.gstart, p0[k] - self.max_size, side="left")
            cover = lo + np.flatnonzero(self.gend[lo : ic[k]] > p0[k])
            inside[k] = True
            if len(cover) == 1:
                ic[k] = cover[0]
            else:
                ambiguous[k] = True
        ok = inside & ~ambiguous
        blk = self.b[ic]
        off = key_position(keys) - 1 - blk[:, 1]
        fwd = blk[:, 5] == 1
        q0 = np.where(fwd, blk[:, 4] + off, blk[:, 4] + blk[:, 2] - 1 - off)  # 0-based on target forward strand
        pos1 = np.where(fwd, q0 + 1, q0)  # minus strand: target C = mapped base - 1 (1-based: q0 + 1 - 1)
        out = (blk[:, 3] << 32) | pos1
        stats = {"n": int(len(keys)), "unmapped": int((~inside).sum()), "ambiguous": int(ambiguous.sum())}
        if target_cpg_keys is not None:
            is_cpg = np.isin(out, target_cpg_keys) & ok
            stats["not_cpg_in_target"] = int((ok & ~is_cpg).sum())
            ok = is_cpg
        stats["kept"] = int(ok.sum())
        return out, ok, stats
