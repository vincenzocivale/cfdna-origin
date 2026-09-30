"""Canonical HDF5 representation artifacts (CpGRepresentationBenchmark contract): `/cpg_idx` int64 locus keys and
`/embedding` [N, D]. Only the key index is held in RAM; embedding rows are read chunk-aligned on demand, so a
genome-wide table (e.g. 29.4M x 256 fp16 = 15 GB) is never loaded as a whole, and never onto the GPU.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from cfdna_origin.data.loci import CODE_CHROM, key_chrom_code
from cfdna_origin.representations.base import (
    LocusEmbeddingStore,
    RepresentationCoverageError,
    RepresentationManifest,
)

ID_KEYS = ("cpg_idx", "cpg_ids", "ids", "id")
EMB_KEYS = ("embedding", "embeddings", "emb", "features")


def _pick(handle: h5py.File, requested: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if requested and requested != "auto":
        if requested not in handle:
            raise RepresentationCoverageError(f"/{requested} absent; datasets: {list(handle.keys())}")
        return requested
    found = [c for c in candidates if c in handle]
    if not found:
        raise RepresentationCoverageError(f"no {kind} dataset among {candidates}; datasets: {list(handle.keys())}")
    return found[0]


def _attrs(obj) -> dict:
    out = {}
    for k, v in obj.attrs.items():
        out[k] = v.item() if isinstance(v, np.generic) else (v.decode() if isinstance(v, bytes) else v)
        if isinstance(out[k], np.ndarray):
            out[k] = out[k].tolist()
    return out


class HDF5LocusEmbeddingStore(LocusEmbeddingStore):
    def __init__(self, path: Path, *, name: str, id_key: str = "auto", embedding_key: str = "auto",
                 normalization: str = "store_sample_standardize", norm_sample_chunks: int = 16, seed: int = 0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"representation artifact for {name!r} not found: {self.path}")
        with h5py.File(self.path, "r") as f:
            self.id_key = _pick(f, id_key, ID_KEYS, "id")
            self.emb_key = _pick(f, embedding_key, EMB_KEYS, "embedding")
            keys = np.asarray(f[self.id_key][:], dtype=np.int64)
            emb = f[self.emb_key]
            if emb.ndim != 2 or emb.shape[0] != len(keys):
                raise RepresentationCoverageError(f"{self.path}: /{self.emb_key} {emb.shape} vs {len(keys)} ids")
            dim, self.block = int(emb.shape[1]), int(emb.chunks[0]) if emb.chunks else 65536
            attrs = {**_attrs(f), **{f"{self.emb_key}.{k}": v for k, v in _attrs(emb).items()}}
            dtype = str(emb.dtype)
        self._sorted = bool(np.all(keys[1:] > keys[:-1]))
        if self._sorted:
            self._order, self._skeys = None, keys
        else:
            self._order = np.argsort(keys, kind="stable")
            self._skeys = keys[self._order]
            if np.any(self._skeys[1:] == self._skeys[:-1]):
                raise RepresentationCoverageError(f"{self.path} has duplicate locus keys")
        codes = np.unique(key_chrom_code(self._skeys[:: max(1, len(keys) // 100_000)]))
        fingerprint = hashlib.sha256()
        fingerprint.update(str((self.path.stat().st_size, len(keys), dim, dtype, sorted(attrs.items()))).encode())
        fingerprint.update(keys[:1_000_000].tobytes()); fingerprint.update(keys[-1_000_000:].tobytes())
        self.manifest = RepresentationManifest(
            name=name, kind="hdf5", dim=dim, source=str(self.path),
            genome_build=str(attrs.get("reference_build", "GRCh38")),
            cpg_universe=f"{len(keys)} loci on {[CODE_CHROM.get(int(c), int(c)) for c in codes]}",
            n_universe_loci=len(keys), preprocessing={k: v for k, v in attrs.items()},
            provenance={"store_fingerprint_sha256": fingerprint.hexdigest(), "dtype": dtype,
                        "id_key": self.id_key, "embedding_key": self.emb_key,
                        "opened_utc": datetime.now(timezone.utc).isoformat()},
        )
        self._mean = np.zeros(dim, np.float32)
        self._scale = np.ones(dim, np.float32)
        if normalization == "store_sample_standardize":
            self._fit_norm(norm_sample_chunks, seed)
        elif normalization != "none":
            raise ValueError(f"unknown normalization {normalization!r}")
        self.manifest.normalization = {"type": normalization, "sample_chunks": norm_sample_chunks, "seed": seed,
                                       "data_derived": False,
                                       "note": "statistics from seeded random chunks of the store itself; "
                                               "no patient or methylation data"}

    def _rows(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.searchsorted(self._skeys, keys)
        ok = pos < len(self._skeys)
        ok[ok] = self._skeys[pos[ok]] == keys[ok]
        rows = pos if self._order is None else self._order[np.minimum(pos, len(self._skeys) - 1)]
        return rows, ok

    def covers(self, keys: np.ndarray) -> np.ndarray:
        return self._rows(np.asarray(keys, dtype=np.int64))[1]

    def _read_rows(self, rows: np.ndarray) -> np.ndarray:
        """Chunk-aligned read of arbitrary rows; returns float32 in the order of `rows`."""
        order = np.argsort(rows, kind="stable")
        srows = rows[order]
        out = np.empty((len(rows), self.dim), np.float32)
        blocks = srows // self.block
        bounds = np.flatnonzero(np.diff(blocks)) + 1
        with h5py.File(self.path, "r") as f:
            emb = f[self.emb_key]
            for seg in np.split(np.arange(len(srows)), bounds):
                if not len(seg):
                    continue
                lo = int(srows[seg[0]]); hi = int(srows[seg[-1]]) + 1
                out[order[seg]] = emb[lo:hi][srows[seg] - lo].astype(np.float32)
        return out

    def _fit_norm(self, n_chunks: int, seed: int) -> None:
        n = len(self._skeys)
        n_blocks = max(1, -(-n // self.block))
        picks = np.sort(np.random.default_rng(seed).choice(n_blocks, min(n_chunks, n_blocks), replace=False))
        s = np.zeros(self.dim, np.float64); ss = np.zeros(self.dim, np.float64); cnt = 0
        with h5py.File(self.path, "r") as f:
            emb = f[self.emb_key]
            for b in picks:
                x = emb[b * self.block : min(n, (b + 1) * self.block)].astype(np.float64)
                s += x.sum(0); ss += (x * x).sum(0); cnt += len(x)
        mean = s / cnt
        std = np.sqrt(np.maximum(ss / cnt - mean ** 2, 0.0))
        self._mean = mean.astype(np.float32)
        self._scale = (1.0 / np.where(std > 1e-6, std, 1.0)).astype(np.float32)

    def reference_sample(self, n: int, seed: int) -> np.ndarray:
        """Seeded random whole chunks of the store (contiguous reads; real CpGs of the store universe)."""
        n_blocks = max(1, -(-len(self._skeys) // self.block))
        rng = np.random.default_rng(seed)
        picks = rng.permutation(n_blocks)[: max(1, -(-n // self.block))]
        rows = np.concatenate([np.arange(b * self.block, min(len(self._skeys), (b + 1) * self.block)) for b in picks])
        rows = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
        return (self._read_rows(rows) - self._mean) * self._scale

    def _lookup(self, keys: np.ndarray) -> np.ndarray:
        rows, _ = self._rows(keys)
        return (self._read_rows(rows) - self._mean) * self._scale
