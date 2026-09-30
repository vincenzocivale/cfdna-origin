#!/usr/bin/env python3
"""Materialise representation tables on a prepared dataset's observed loci (optional: train.py does it lazily).

  python scripts/materialize_representation.py --experiment gse149438_main [--representations functional random]
Fails explicitly if an artifact is missing or a dataset locus is outside a representation's universe.
"""
from __future__ import annotations

import argparse
import json

from cfdna_origin.config import load_experiment, load_paths
from cfdna_origin.experiments.runner import load_dataset, representation_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--representations", nargs="*")
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    paths = load_paths()
    store, _, _ = load_dataset(exp["dataset"], paths)
    for rep in args.representations or exp["representations"]:
        table, manifest = representation_table(exp, rep, store, paths)
        print(rep, table.shape, json.dumps(manifest.get("projection")))


if __name__ == "__main__":
    main()
