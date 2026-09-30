"""wgbstools-style global CpG index -> hg38 coordinates.

PAT files address CpGs by a 1-based index over all CpGs of the reference, ordered chr1..chr22, chrX, chrY, chrM
(strands collapsed onto the C of the forward strand). The order is rebuilt from the FASTA and checked against
the known totals (hg38 .beta files hold one uint8 pair per CpG: 29,401,795 CpGs).
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
EXPECTED_TOTAL = 29_401_795  # 58_803_590 bytes / 2 for a hg38 .beta


def _read_fasta(path: Path, wanted: set[str]):
    opener = gzip.open if str(path).endswith(".gz") else open
    name, parts = None, []
    with opener(path, "rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                if name in wanted:
                    yield name, b"".join(parts)
                name, parts = line[1:].split()[0].decode(), []
            elif name in wanted:
                parts.append(line.strip())
    if name in wanted:
        yield name, b"".join(parts)


def cpg_positions(seq: bytes) -> np.ndarray:
    """1-based positions of every C followed by G (soft-masked bases upper-cased)."""
    arr = np.frombuffer(seq, dtype=np.uint8) & 0xDF
    return (np.flatnonzero((arr[:-1] == ord("C")) & (arr[1:] == ord("G"))) + 1).astype(np.int32)


def build_index(fasta: Path) -> dict[str, np.ndarray]:
    found = dict(_read_fasta(fasta, set(CHROMS)))
    missing = [c for c in CHROMS if c not in found]
    if missing:
        raise ValueError(f"FASTA lacks contigs {missing}")
    return {c: cpg_positions(found[c]) for c in CHROMS}


def save_index(index: dict[str, np.ndarray], out: Path) -> None:
    sizes = np.array([len(index[c]) for c in CHROMS], dtype=np.int64)
    np.savez(
        out,
        chroms=np.array(CHROMS),
        offsets=np.concatenate([[0], np.cumsum(sizes)]),  # global 0-based start of each chrom
        positions=np.concatenate([index[c] for c in CHROMS]),
    )


class CpGIndex:
    """Lookup from (chrom, 1-based global CpG index) to genomic position."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.chroms = [str(c) for c in z["chroms"]]
        self.offsets = z["offsets"]
        self.positions = z["positions"]
        self._chrom_row = {c: i for i, c in enumerate(self.chroms)}

    @property
    def total(self) -> int:
        return int(self.offsets[-1])

    def chrom_of(self, global_idx: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.offsets, np.asarray(global_idx) - 1, side="right") - 1

    def position(self, global_idx: np.ndarray) -> np.ndarray:
        return self.positions[np.asarray(global_idx, dtype=np.int64) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    index = build_index(args.fasta)
    total = sum(len(v) for v in index.values())
    autosomal = sum(len(index[c]) for c in CHROMS[:22])
    print(f"total CpGs {total} (expected {EXPECTED_TOTAL}), autosomal {autosomal}")
    if total != EXPECTED_TOTAL:
        raise SystemExit("CpG total differs from the hg38 wgbstools index; refusing to write")
    save_index(index, args.out)


if __name__ == "__main__":
    main()
