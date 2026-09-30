"""Reference CpG sets from a FASTA (used to verify lifted/extracted loci are CpGs of the target assembly)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cfdna_origin.data.loci import AUTOSOMES, make_keys


def cpg_positions(seq: bytes) -> np.ndarray:
    """1-based positions of every C followed by G (soft-masked bases upper-cased)."""
    arr = np.frombuffer(seq, dtype=np.uint8) & 0xDF
    return (np.flatnonzero((arr[:-1] == ord("C")) & (arr[1:] == ord("G"))) + 1).astype(np.int64)


def read_fasta(path: Path, wanted: set[str]):
    """Stream (name, sequence) for the wanted contigs of a plain or gzip FASTA (no index needed)."""
    import gzip

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


def reference_cpg_keys(fasta: Path, cache: Path, contigs: list[str] | None = None) -> np.ndarray:
    """Sorted locus keys of all CpGs on `contigs` (default autosomes); cached as .npy."""
    if cache.exists():
        return np.load(cache)
    wanted = set(contigs or AUTOSOMES)
    keys = [make_keys(name, cpg_positions(seq)) for name, seq in read_fasta(fasta, wanted)]
    if len(keys) != len(wanted):
        raise ValueError(f"{fasta} lacks some of the contigs {sorted(wanted)}")
    out = np.sort(np.concatenate(keys))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out
