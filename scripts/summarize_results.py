#!/usr/bin/env python3
"""Aggregate completed runs of an experiment: per-seed / per-arm metrics and paired patient-level comparisons.

  python scripts/summarize_results.py --experiment gse149438_main
Writes {outputs_root}/<dataset>/<experiment>/summary/{runs,per_seed,arms,comparisons}.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from cfdna_origin.config import load_experiment, load_paths
from cfdna_origin.experiments.runner import load_dataset
from cfdna_origin.evaluation.summary import summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--subset", default="all", help="eval subset: all | heldout_loci_only")
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    paths = load_paths()
    _, _, classes = load_dataset(exp["dataset"], paths)
    exp_dir = paths["outputs_root"] / exp["dataset"]["name"] / exp["name"]
    b = exp.get("bootstrap", {})
    res = summarize(exp_dir, classes, exp["seeds"], exp["folds"], [tuple(p) for p in exp.get("comparisons", [])],
                    n_boot=b.get("n_boot", 2000), n_perm=b.get("n_perm", 2000), subset=args.subset)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(res["arms"]); print(res["comparisons"])
    if res["incomplete"]:
        print("incomplete arms (excluded from comparisons):", res["incomplete"])
    print("written to", exp_dir / "summary")


if __name__ == "__main__":
    main()
