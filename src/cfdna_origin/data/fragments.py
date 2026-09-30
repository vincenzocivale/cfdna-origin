"""On-disk fragment store and bag construction (CPU side of the pipeline).

Processed dataset layout (`<processed_dir>/`):
    samples.parquet              one row per sample (see `schema.SAMPLE_COLUMNS`)
    loci.npy                     int64 sorted union of observed locus keys (coordinates only, label-free)
    dataset_manifest.json        build parameters, counts, filters, sha256 of loci.npy
    fragments/<sample_id>/
        frag_offsets.npy         int64 [F+1]  CSR offsets into the CpG arrays
        state.npy                uint8 [C]    1 = methylated, 0 = unmethylated
        locus_row.npy            int32 [C]    row in loci.npy (written by `finalize_dataset`)
        locus_key.npy            int64 [C]    GRCh38 locus keys, position-sorted within each fragment; written by the
                                              extractor and removed by `finalize_dataset` (keys = loci[locus_row]),
                                              which cuts the store from 13 to 5 bytes per CpG call

A fragment (read pair / read) is the set of its called CpGs. All arrays are memory-mapped; nothing is loaded whole.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from cfdna_origin.data.loci import key_position


def write_sample(out_dir: Path, frag_offsets: np.ndarray, locus_key: np.ndarray, state: np.ndarray) -> dict:
    frag_offsets = np.asarray(frag_offsets, np.int64)
    locus_key = np.asarray(locus_key, np.int64); state = np.asarray(state, np.uint8)
    if frag_offsets[0] != 0 or frag_offsets[-1] != len(locus_key) or len(state) != len(locus_key):
        raise ValueError("inconsistent CSR arrays")
    if np.any(np.diff(frag_offsets) <= 0):
        raise ValueError("empty fragments are not allowed")
    if not np.isin(state, (0, 1)).all():
        raise ValueError("state must be 0/1")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("frag_offsets", frag_offsets), ("locus_key", locus_key), ("state", state)):
        np.save(out_dir / f"{name}.tmp.npy", arr)
        (out_dir / f"{name}.tmp.npy").replace(out_dir / f"{name}.npy")
    return {"n_fragments": int(len(frag_offsets) - 1), "n_cpgs": int(len(locus_key))}


def filter_calls(frag_offsets: np.ndarray, locus_key: np.ndarray, state: np.ndarray, keep: np.ndarray):
    """Drop CpG calls where `keep` is False; fragments left without calls are removed. Returns CSR arrays + counts."""
    frag_of = np.repeat(np.arange(len(frag_offsets) - 1), np.diff(frag_offsets))[keep]
    counts = np.bincount(frag_of, minlength=len(frag_offsets) - 1)
    new_off = np.concatenate([[0], np.cumsum(counts[counts > 0])]).astype(np.int64)
    return new_off, locus_key[keep], state[keep], {"calls_dropped": int((~keep).sum()),
                                                   "fragments_emptied": int((counts == 0).sum())}


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sample_keys(d: Path, old_loci: np.ndarray | None) -> np.ndarray:
    if (d / "locus_key.npy").exists():
        return np.load(d / "locus_key.npy", mmap_mode="r")
    if old_loci is None or not (d / "locus_row.npy").exists():
        raise FileNotFoundError(f"{d}: neither locus_key.npy nor (locus_row.npy + previous loci.npy)")
    return old_loci[np.load(d / "locus_row.npy")]


def finalize_dataset(processed_dir: Path, sample_ids: list[str], build_info: dict, drop_locus_key: bool = True) -> dict:
    """Union of observed loci -> loci.npy; per-sample locus_row.npy; dataset_manifest.json.

    Re-runnable (e.g. a partial build extended later): samples finalized before are re-keyed from the previous
    loci.npy. New files are written next to the old ones and swapped in only when all are ready."""
    frag_root = processed_dir / "fragments"
    old_loci = np.load(processed_dir / "loci.npy") if (processed_dir / "loci.npy").exists() else None
    loci = np.unique(np.concatenate([np.unique(_sample_keys(frag_root / s, old_loci)) for s in sample_ids]))
    for s in sample_ids:
        np.save(frag_root / s / "locus_row.new.npy",
                np.searchsorted(loci, _sample_keys(frag_root / s, old_loci)).astype(np.int32))
    np.save(processed_dir / "loci.new.npy", loci)
    (processed_dir / "loci.new.npy").replace(processed_dir / "loci.npy")
    for s in sample_ids:
        (frag_root / s / "locus_row.new.npy").replace(frag_root / s / "locus_row.npy")
        if drop_locus_key:
            (frag_root / s / "locus_key.npy").unlink(missing_ok=True)
    manifest = {**build_info, "n_samples": len(sample_ids), "n_loci": int(len(loci)),
                "loci_sha256": sha256_array(loci)}
    (processed_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


@dataclasses.dataclass
class Bag:
    """Padded batch of fragments. rows/state/geom/mask are [F, L(, 3)]; `sample` indexes the owning sample."""
    rows: np.ndarray  # int64 dataset locus rows
    state: np.ndarray  # int64 0/1
    geom: np.ndarray  # float32 [F, L, 3]: log1p(offset from first CpG)/7, log1p(gap to previous)/7, rank
    mask: np.ndarray  # bool, True = real CpG
    sample: np.ndarray  # int64 [F] sample index within the batch

    def __len__(self) -> int:
        return len(self.rows)


class FragmentStore:
    """Read-only access to the fragments of one processed dataset.

    Filters (applied identically for every representation arm): fragments with fewer than `min_cpgs` called CpGs are
    invalid; fragments longer than `max_cpgs` are truncated to their first `max_cpgs` CpGs (position order).
    Optionally, fragments touching loci in `heldout_loci` (bool mask over loci.npy) can be excluded (unseen-locus
    protocol).
    """

    def __init__(self, processed_dir: Path, *, min_cpgs: int = 3, max_cpgs: int = 32, cache_samples: int = 64):
        self.dir = Path(processed_dir)
        self.manifest = json.loads((self.dir / "dataset_manifest.json").read_text())
        self.loci = np.load(self.dir / "loci.npy", mmap_mode="r")
        if len(self.loci) != self.manifest["n_loci"]:
            raise ValueError("loci.npy does not match dataset_manifest.json")
        self.min_cpgs, self.max_cpgs = int(min_cpgs), int(max_cpgs)
        self._load = lru_cache(maxsize=cache_samples)(self._load_uncached)

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256((self.dir / "dataset_manifest.json").read_bytes()).hexdigest()

    def _load_uncached(self, sample_id: str) -> dict:
        d = self.dir / "fragments" / sample_id
        if not (d / "locus_row.npy").exists():
            raise FileNotFoundError(f"{d} missing or not finalized (run the dataset preparation to completion)")
        off = np.load(d / "frag_offsets.npy")
        n = np.diff(off)
        return {"off": off, "row": np.load(d / "locus_row.npy", mmap_mode="r"),
                "state": np.load(d / "state.npy", mmap_mode="r"),
                "valid": np.flatnonzero(n >= self.min_cpgs), "n_cpg": n}

    def n_fragments(self, sample_id: str) -> int:
        return len(self._load(sample_id)["off"]) - 1

    def valid_fragments(self, sample_id: str, heldout_loci: np.ndarray | None = None,
                        mode: str = "exclude") -> np.ndarray:
        """Indices of valid fragments. With `heldout_loci`: mode 'exclude' drops fragments touching held-out loci,
        'only' keeps only those fragments."""
        s = self._load(sample_id)
        valid = s["valid"]
        if heldout_loci is None:
            return valid
        touched = np.zeros(len(s["off"]) - 1, dtype=bool)
        hit = heldout_loci[np.asarray(s["row"])]
        np.logical_or.at(touched, np.repeat(np.arange(len(touched)), s["n_cpg"])[hit], True)
        keep = touched[valid] if mode == "only" else ~touched[valid]
        return valid[keep]

    def sample_stats(self, sample_id: str) -> dict:
        s = self._load(sample_id)
        v = s["valid"]
        return {"number_of_reads": int(len(s["off"]) - 1), "number_of_valid_reads": int(len(v)),
                "mean_CpGs_per_read": float(np.minimum(s["n_cpg"][v], self.max_cpgs).mean()) if len(v) else 0.0}

    def bag(self, sample_ids: list[str], fragment_idx: list[np.ndarray]) -> Bag:
        """Assemble the given fragments (per sample) into one padded Bag."""
        lens, starts, owners = [], [], []
        for i, (sid, idx) in enumerate(zip(sample_ids, fragment_idx)):
            s = self._load(sid)
            starts.append(s["off"][idx]); lens.append(np.minimum(s["n_cpg"][idx], self.max_cpgs))
            owners.append(np.full(len(idx), i, np.int64))
        n = np.concatenate(lens) if lens else np.zeros(0, np.int64)
        L = int(n.max()) if len(n) else 1
        F = len(n)
        rows = np.zeros((F, L), np.int64); state = np.zeros((F, L), np.int64); pos = np.zeros((F, L), np.int64)
        mask = np.arange(L)[None] < n[:, None]
        at = 0
        for sid, st, ln in zip(sample_ids, starts, lens):
            s = self._load(sid)
            k = len(st)
            if not k:
                continue
            flat = (st[:, None] + np.arange(L)[None])[mask[at : at + k]]
            m = mask[at : at + k]
            rows[at : at + k][m] = s["row"][flat]
            state[at : at + k][m] = s["state"][flat]
            pos[at : at + k][m] = key_position(self.loci[np.asarray(s["row"][flat])])
            at += k
        first = pos[:, :1]
        prev = np.concatenate([pos[:, :1], pos[:, :-1]], 1)
        rank = np.arange(L)[None] / np.maximum(n[:, None] - 1, 1)
        geom = np.stack([np.log1p(np.clip(pos - first, 0, None)) / 7.0,
                         np.log1p(np.clip(pos - prev, 0, None)) / 7.0, rank], -1)
        geom = np.where(mask[..., None], geom, 0.0).astype(np.float32)
        return Bag(rows=rows, state=state, geom=geom, mask=mask, sample=np.concatenate(owners) if owners else
                   np.zeros(0, np.int64))


def heldout_locus_mask(loci: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Deterministic hash partition of loci (depends only on the locus key and seed, not on the data)."""
    from cfdna_origin.representations.controls import _splitmix64, _uniform  # local import: shared hash

    if fraction <= 0:
        return np.zeros(len(loci), dtype=bool)
    u = _uniform(_splitmix64(np.asarray(loci, np.int64).astype(np.uint64) ^ np.uint64(seed * 7919 + 1)))
    return u < fraction
