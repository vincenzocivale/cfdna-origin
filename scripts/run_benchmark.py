#!/usr/bin/env python3
"""Run every (representation x seed x fold) of an experiment sequentially; completed runs are skipped.

  python scripts/run_benchmark.py --experiment gse149438_main [--representations functional methylation_only]
                                  [--seeds 17] [--folds 0] [--device cuda:0]
Unavailable representation artifacts (e.g. ntv3_pre not yet materialised) fail explicitly unless --skip-unavailable.
"""
from __future__ import annotations

import argparse
import traceback

from cfdna_origin.config import load_experiment
from cfdna_origin.experiments.runner import run_one
from cfdna_origin.representations.registry import RepresentationUnavailableError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--representations", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--folds", nargs="*", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-unavailable", action="store_true")
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    reps = args.representations or list(exp["representations"])
    unknown = set(reps) - set(exp["representations"])
    if unknown:
        raise SystemExit(f"representations not in the experiment: {sorted(unknown)}")
    failed = []
    # fold-major, seed, then representation: arms of the same (fold, seed) finish close in time
    for fold in args.folds if args.folds is not None else exp["folds"]:
        for seed in args.seeds or exp["seeds"]:
            for rep in reps:
                try:
                    run_one(exp, rep, seed, fold, device=args.device)
                except RepresentationUnavailableError as e:
                    if not args.skip_unavailable:
                        raise
                    print(f"SKIP {rep}: {e}")
                except Exception:
                    failed.append((rep, seed, fold)); traceback.print_exc()
    if failed:
        raise SystemExit(f"failed runs: {failed}")


if __name__ == "__main__":
    main()
