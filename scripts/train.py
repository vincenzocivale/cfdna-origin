#!/usr/bin/env python3
import argparse
from pathlib import Path

from cfdna_too.training.run import run

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["functional", "random", "none"], required=True)
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--d-model", type=int, default=192)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--eval-reads", type=int, default=100000)
ap.add_argument("--embedding-h5", default="data/functional_pca_genomewide_hg38.h5")
a = ap.parse_args()
cfg = dict(root=str(ROOT), splits=str(ROOT / "data/meta/splits.parquet"), reads=str(ROOT / "data/reads"), out=str(ROOT / "outputs"),
           arm=a.arm, seed=a.seed, steps=a.steps, batch=a.batch, lr=a.lr, d_model=a.d_model, n_layers=a.n_layers,
           eval_every=a.eval_every, eval_reads=a.eval_reads, embedding_h5=a.embedding_h5)
r = run(cfg)
print({k: v for k, v in r.items() if k in ("arm", "seed", "best_val_read_macro_f1", "test")})
