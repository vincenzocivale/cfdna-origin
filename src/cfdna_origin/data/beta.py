"""Read access to a processed-beta dataset (dense beta / coverage matrices over GRCh38 loci)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class BetaData:
    """Observed tokens per sample: (locus row, beta, coverage). Missing CpGs are never materialised as tokens."""

    def __init__(self, processed_dir: Path, in_memory: bool = True):
        self.dir = Path(processed_dir)
        mpath = self.dir / "dataset_manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"{mpath} missing: run scripts/prepare_gse149438_beta.py")
        self.manifest = json.loads(mpath.read_text())
        self.loci = np.load(self.dir / "loci.npy")
        self.loci_hg19 = np.load(self.dir / "loci_hg19.npy")
        mode = None if in_memory else "r"
        self.beta = np.load(self.dir / "beta.npy", mmap_mode=mode)
        self.cov = np.load(self.dir / "coverage.npy", mmap_mode=mode)
        self.samples = pd.read_parquet(self.dir / "samples.parquet")
        if self.beta.shape != (len(self.samples), len(self.loci)) or self.cov.shape != self.beta.shape:
            raise ValueError("matrix shapes do not match samples/loci")
        self.row_of = {s: i for i, s in enumerate(self.samples.sample_id)}
        self._obs: dict[str, np.ndarray] = {}

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256((self.dir / "dataset_manifest.json").read_bytes()).hexdigest()

    def observed(self, sample_id: str) -> np.ndarray:
        if sample_id not in self._obs:
            self._obs[sample_id] = np.flatnonzero(np.asarray(self.cov[self.row_of[sample_id]]) > 0).astype(np.int64)
        return self._obs[sample_id]

    def tokens(self, sample_id: str, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.row_of[sample_id]
        return cols, np.asarray(self.beta[r, cols], np.float32), np.asarray(self.cov[r, cols], np.float32)

    def summary_features(self, sample_ids: list[str]) -> pd.DataFrame:
        cols = ["n_observed_loci", "mean_beta", "beta_variance", "mean_coverage", "median_coverage", "missing_fraction"]
        return self.samples.set_index("sample_id").loc[sample_ids, cols]
