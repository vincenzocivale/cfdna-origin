#!/usr/bin/env python3
"""Recompute (and print) the metrics of a completed run from its saved predictions — no model re-execution, so the
test set is never re-evaluated. Use --check to fail if they differ from the stored metrics.json.

  python scripts/evaluate.py <run_dir> [--check]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfdna_origin.evaluation.metrics import all_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (args.run_dir / "RUN_COMPLETE.json").exists():
        raise SystemExit(f"{args.run_dir} is not a completed run")
    stored = json.loads((args.run_dir / "metrics.json").read_text())
    classes = stored["test/all"]["confusion_matrix"]["labels"]
    pred = pd.read_parquet(args.run_dir / "predictions.parquet")
    idx = {c: i for i, c in enumerate(classes)}
    for (split, subset), g in pred.groupby(["split", "eval_subset"]):
        m = all_metrics(g.true_label.map(idx).to_numpy(), g[[f"prob_{c}" for c in classes]].to_numpy(), classes)
        print(f"{split}/{subset}: " + ", ".join(f"{k}={v:.4f}" for k, v in m["primary"].items()))
        if args.check:
            old = stored[f"{split}/{subset}"]["primary"]
            for k, v in m["primary"].items():
                if not np.isclose(v, old[k], equal_nan=True):
                    raise SystemExit(f"mismatch {split}/{subset} {k}: {v} vs stored {old[k]}")
    if args.check:
        print("metrics match metrics.json")


if __name__ == "__main__":
    main()
