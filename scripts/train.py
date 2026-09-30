#!/usr/bin/env python3
"""Train + evaluate ONE run (experiment, representation, seed, fold). Writes the full run directory.

  python scripts/train.py --experiment gse149438_main --representation functional --seed 17 --fold 0
"""
from __future__ import annotations

import argparse

from cfdna_origin.config import load_experiment
from cfdna_origin.experiments.runner import run_one


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="name in configs/experiments or a YAML path")
    ap.add_argument("--representation", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="re-run even if RUN_COMPLETE.json exists")
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    if args.representation not in exp["representations"]:
        raise SystemExit(f"{args.representation!r} not in experiment representations {list(exp['representations'])}")
    run_one(exp, args.representation, args.seed, args.fold, device=args.device, force=args.force)


if __name__ == "__main__":
    main()
