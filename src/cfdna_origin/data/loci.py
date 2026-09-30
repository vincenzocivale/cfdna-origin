"""CpG locus keys: `chrom_code << 32 | 1-based position of the CpG cytosine (forward strand)`, GRCh38.

Same namespace as CpGRepresentationBenchmark (`locus_key_chromcode<<32|pos_v1`): chr1..chr22 -> 1..22, chrX -> 23,
chrY -> 24. Reads on the reverse strand are collapsed onto the forward-strand C (reverse-strand G position - 1).
"""
from __future__ import annotations

import numpy as np

CHROM_CODE = {**{f"chr{i}": i for i in range(1, 23)}, "chrX": 23, "chrY": 24}
CODE_CHROM = {v: k for k, v in CHROM_CODE.items()}
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
GENOME_BUILD = "GRCh38"
KEY_NAMESPACE = "locus_key_chromcode<<32|pos_v1"
_MASK = np.int64(0xFFFFFFFF)


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def make_keys(chrom: str | np.ndarray, pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.int64)
    if np.any(pos <= 0) or np.any(pos > _MASK):
        raise ValueError("positions must be 1-based and < 2**32")
    if isinstance(chrom, str):
        code = np.full(pos.shape, CHROM_CODE[normalize_chrom(chrom)], dtype=np.int64)
    else:
        code = np.asarray([CHROM_CODE[normalize_chrom(c)] for c in chrom], dtype=np.int64)
    return (code << np.int64(32)) | pos


def key_chrom_code(keys: np.ndarray) -> np.ndarray:
    return np.asarray(keys, dtype=np.int64) >> np.int64(32)


def key_position(keys: np.ndarray) -> np.ndarray:
    return np.asarray(keys, dtype=np.int64) & _MASK


def chrom_codes(chroms: list[str]) -> np.ndarray:
    return np.asarray([CHROM_CODE[normalize_chrom(c)] for c in chroms], dtype=np.int64)


# GRCh38 primary assembly chromosome lengths (bp), used only to draw label-free reference coordinates.
GRCH38_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
    "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
    "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}


def random_genome_keys(n: int, seed: int, chroms: list[str] | None = None) -> np.ndarray:
    """Seeded uniform genomic coordinates (not necessarily CpGs), length-weighted over `chroms` (default autosomes)."""
    chroms = chroms or AUTOSOMES
    lengths = np.asarray([GRCH38_LENGTHS[c] for c in chroms], dtype=np.float64)
    rng = np.random.default_rng(seed)
    which = rng.choice(len(chroms), n, p=lengths / lengths.sum())
    pos = (rng.random(n) * lengths[which]).astype(np.int64) + 1
    return np.unique((chrom_codes(chroms)[which] << np.int64(32)) | pos)
