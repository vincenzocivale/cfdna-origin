"""Shared tiny synthetic fixtures (unit tests only; every file is a few KB and lives under pytest's tmp_path)."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from cfdna_origin.data.fragments import finalize_dataset, write_sample
from cfdna_origin.data.loci import make_keys

torch.set_num_threads(1)

CLASSES = ["Healthy", "Colorectal", "Gastric"]
DIAG = {"Healthy": "Normal", "Colorectal": "CRC", "Gastric": "GC"}


def write_h5(path, keys, emb, chunks=None, attrs=None):
    with h5py.File(path, "w") as f:
        f["cpg_idx"] = np.asarray(keys, np.int64)
        f.create_dataset("embedding", data=np.asarray(emb), chunks=chunks)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    return path


def make_dataset(root, *, n_patients_per_class=5, n_fragments=40, seed=0):
    """Processed dataset under `root` (samples.parquet, loci.npy, fragments/); one sample per patient."""
    rng = np.random.default_rng(seed)
    universe = np.concatenate([make_keys("chr1", np.arange(100, 100 + 80 * 23, 23)),
                               make_keys("chr2", np.arange(500, 500 + 40 * 31, 31))])
    rows = []
    for c in CLASSES:
        for p in range(n_patients_per_class):
            sid = f"S_{c[:3]}_{p}"
            offs, keys, states = [0], [], []
            for _ in range(n_fragments):
                start = rng.integers(0, len(universe) - 8)
                k = universe[start : start + rng.integers(1, 8)]
                keys.append(k); offs.append(offs[-1] + len(k))
                states.append(rng.integers(0, 2, len(k)).astype(np.uint8))
            res = write_sample(root / "fragments" / sid, np.asarray(offs), np.concatenate(keys), np.concatenate(states))
            rows.append({"sample_id": sid, "patient_id": f"P_{sid}", "diagnosis": DIAG[c], "label": c,
                         "label_idx": CLASSES.index(c), "sample_type": "plasma_cfdna", "batch": f"b{p % 2}", **res})
    samples = pd.DataFrame(rows)
    finalize_dataset(root, samples.sample_id.tolist(), {"dataset": "tiny"})
    samples.to_parquet(root / "samples.parquet", index=False)
    return samples


@pytest.fixture()
def tiny_dataset(tmp_path):
    root = tmp_path / "processed"
    return root, make_dataset(root)


@pytest.fixture()
def tiny_samples():
    """Sample table only (no fragments): 3 classes x 10 patients, 2 batches, one sample per patient."""
    rows = []
    for c in CLASSES:
        for p in range(10):
            rows.append({"sample_id": f"S_{c}_{p}", "patient_id": f"P_{c}_{p}", "diagnosis": DIAG[c], "label": c,
                         "label_idx": CLASSES.index(c), "batch": f"b{p % 2}"})
    return pd.DataFrame(rows)
