"""Build a LocusEmbeddingStore from a `configs/representations/*.yaml` entry and materialise it on a dataset's
observed loci.

Materialisation writes `<cache>/<dataset>/<representation>/embedding.npy` (float16, row i = dataset locus i of
`loci.npy`) plus `manifest.json`. Training only ever reads this per-dataset table (memory-mapped on CPU, gathered per
batch), so the genome-wide artifact is read once per (dataset, representation) and never duplicated whole.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cfdna_origin.config import resolve_path
from cfdna_origin.representations.base import LocusEmbeddingStore, NoLocusStore
from cfdna_origin.representations.controls import PositionOnlyStore, RandomLocusEmbeddingStore
from cfdna_origin.representations.hdf5 import HDF5LocusEmbeddingStore


class RepresentationUnavailableError(FileNotFoundError):
    pass


def build_store(cfg: dict, paths: dict[str, Path]) -> LocusEmbeddingStore:
    kind, name = cfg["kind"], cfg["name"]
    if kind == "none":
        return NoLocusStore(name)
    if kind == "random":
        return RandomLocusEmbeddingStore(dim=cfg.get("dim", 256), seed=cfg.get("seed", 0), name=name)
    if kind == "position":
        return PositionOnlyStore(dim=cfg.get("dim", 256), name=name)
    if kind == "hdf5":
        path = resolve_path(cfg["path"], paths)
        if not path.exists():
            raise RepresentationUnavailableError(
                f"representation {name!r}: artifact {path} not found. {cfg.get('availability', '')}".strip())
        return HDF5LocusEmbeddingStore(path, name=name, id_key=cfg.get("id_key", "auto"),
                                       embedding_key=cfg.get("embedding_key", "auto"),
                                       normalization=cfg.get("normalization", "store_sample_standardize"),
                                       norm_sample_chunks=cfg.get("norm_sample_chunks", 16), seed=cfg.get("norm_seed", 0))
    raise ValueError(f"unknown representation kind {kind!r}")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def fit_projection(store: LocusEmbeddingStore, spec: dict | None) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Common, label-free projection applied to every arm (experiment-level `representation_projection`).

    `pca`: centre + top-r principal axes fitted on a seeded genome-wide sample of the store itself (no dataset loci,
    no patients), then one global rescaling to unit mean variance. Use `normalization: none` on the store so the PCA
    sees the artifact's native variance structure (per-dimension z-scoring first would flatten it). Every arm with a locus embedding then enters the model with the same width r, so the trainable
    input adapter has identical size across arms."""
    if not spec or spec.get("type", "none") == "none" or store.dim == 0:
        return None, None, {"type": "none"}
    if spec["type"] != "pca":
        raise ValueError(f"unknown projection {spec['type']!r}")
    r = int(spec["dim"])
    if store.dim < r:
        raise ValueError(f"{store.name}: dim {store.dim} < projection dim {r}")
    x = store.reference_sample(int(spec.get("fit_loci", 200_000)), int(spec.get("seed", 0))).astype(np.float64)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    evr = (s ** 2) / (s ** 2).sum()
    # one global scale (not per dimension) so the retained components have unit mean variance: keeps the variance
    # ordering of the representation while putting every arm on the same input scale
    scale = 1.0 / np.sqrt(np.mean(s[:r] ** 2) / max(len(x) - 1, 1))
    comps = (vt[:r].T * scale).astype(np.float32)
    info = {"type": "pca", "dim": r, "fit_loci": int(len(x)), "seed": int(spec.get("seed", 0)), "global_scale": float(scale),
            "explained_variance_ratio": float(evr[:r].sum()), "data_derived": False,
            "components_sha256": hashlib.sha256(comps.tobytes()).hexdigest()}
    return mean.astype(np.float32), comps, info


def projection_tag(spec: dict | None) -> str:
    return "raw" if not spec or spec.get("type", "none") == "none" else f"{spec['type']}{spec['dim']}"


def materialize(store: LocusEmbeddingStore, loci: np.ndarray, out_dir: Path, *, dataset_manifest_sha: str,
                projection: dict | None = None, shuffle_loci_seed: int | None = None,
                batch: int = 2_000_000) -> dict:
    """Gather (and project) embeddings for the sorted dataset loci; fails if any locus is outside the store universe.

    `shuffle_loci_seed` (control arm): permute the materialised rows across the dataset loci — same marginal
    distribution of vectors, locus-to-content link destroyed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if np.any(np.diff(loci) <= 0):
        raise ValueError("dataset loci must be strictly increasing")
    store.get_embeddings(loci[:0])  # interface check
    covered = store.covers(loci)
    if not covered.all():
        store.get_embeddings(loci)  # raises RepresentationCoverageError with details
    mean, comps, proj_info = fit_projection(store, projection)
    width = store.dim if comps is None else comps.shape[1]
    tmp = out_dir / "embedding.tmp.npy"
    table = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(loci), width))
    for s in range(0, len(loci), batch):
        x = store.get_embeddings(loci[s : s + batch])
        if comps is not None:
            x = (x - mean) @ comps
        table[s : s + batch] = x.astype(np.float16)
    if shuffle_loci_seed is not None and len(loci):
        table[:] = table[np.random.default_rng(shuffle_loci_seed).permutation(len(loci))]
    table.flush(); del table
    tmp.replace(out_dir / "embedding.npy")
    manifest = store.manifest.to_dict()
    manifest["dim_materialized"] = int(width)
    manifest["projection"] = proj_info
    manifest["shuffle_loci_seed"] = shuffle_loci_seed
    manifest["materialization"] = {
        "n_loci": int(len(loci)), "dtype": "float16", "dataset_manifest_sha256": dataset_manifest_sha,
        "table_sha256": sha256_file(out_dir / "embedding.npy"), "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def load_materialized(out_dir: Path, expected_n_loci: int, dataset_manifest_sha: str) -> tuple[np.ndarray, dict]:
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        raise RepresentationUnavailableError(
            f"{out_dir} is not materialised; run scripts/materialize_representation.py first")
    manifest = json.loads(mpath.read_text())
    mat = manifest["materialization"]
    if mat["n_loci"] != expected_n_loci or mat["dataset_manifest_sha256"] != dataset_manifest_sha:
        raise ValueError(f"{out_dir} was materialised for a different dataset build; re-materialise")
    return np.load(out_dir / "embedding.npy", mmap_mode="r"), manifest
