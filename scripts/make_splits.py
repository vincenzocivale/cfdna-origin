#!/usr/bin/env python3
"""Donor-group split of the reference samples (train/val/test) + external sets (real cfDNA / matched WBC).

Assignment is global at donor-group level (a group never appears in two splits, even across organs). A seeded search
maximises the number of organs that have >=1 train AND >=1 test group (and a val group when they have >=3).
Organs with a single donor group cannot be tested on unseen donors: they stay train-only and are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def score(assign: dict[str, str], groups_of_organ: dict[str, set[str]]) -> float:
    s = 0.0
    for organ, gs in groups_of_organ.items():
        sp = [assign[g] for g in gs]
        if len(gs) >= 2 and "train" in sp and "test" in sp:
            s += 1.0
            if len(gs) >= 3 and "val" in sp:
                s += 0.5
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, default=ROOT / "data/meta/samples.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/meta/splits.parquet")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    df = pd.read_parquet(a.samples)
    ref = df[df.role == "reference"]
    groups = sorted(ref.donor_group.unique())
    organs = {o: set(g.donor_group) for o, g in ref.groupby("organ")}
    rng = np.random.default_rng(a.seed)
    n = len(groups)
    best, best_s = None, -1.0
    for _ in range(a.trials):
        perm = rng.permutation(n)
        nt, nv = int(round(a.test_frac * n)), int(round(a.val_frac * n))
        assign = {groups[i]: "train" for i in perm}
        for i in perm[:nt]:
            assign[groups[i]] = "test"
        for i in perm[nt : nt + nv]:
            assign[groups[i]] = "val"
        s = score(assign, organs)
        if s > best_s:
            best, best_s = assign, s
    df["split"] = df.donor_group.map(best)
    df.loc[df.role == "cfdna", "split"] = "external_cfdna"
    df.loc[df.role == "wbc", "split"] = "external_wbc"
    assert df.split.notna().all()
    # a donor group appears in exactly one reference split
    assert (ref.assign(s=ref.donor_group.map(best)).groupby("donor_group").s.nunique() == 1).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    r = df[df.role == "reference"]
    testable = [o for o, g in organs.items() if len(g) >= 2]
    ok = [o for o in testable if {"train", "test"} <= set(r[r.organ == o].split)]
    report = {"seed": a.seed, "trials": a.trials, "score": best_s,
              "samples_per_split": df.split.value_counts().to_dict(),
              "organs_total": len(organs), "organs_with_>=2_groups": len(testable), "organs_train_and_test": len(ok),
              "organs_train_only_single_group": sorted(o for o, g in organs.items() if len(g) < 2),
              "organs_testable_but_uncovered": sorted(set(testable) - set(ok)),
              "sha256_assignment": hashlib.sha256(json.dumps(sorted(best.items())).encode()).hexdigest()}
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
