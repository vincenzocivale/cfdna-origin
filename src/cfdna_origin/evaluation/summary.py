"""Collect completed runs of one experiment and produce the benchmark tables.

- runs.csv:        one row per completed run (primary test metrics per fold)
- per_seed.csv:    test predictions pooled over folds (each patient predicted once per seed) -> metrics per seed
- arms.csv:        mean ± sd over seeds, plus the seed-ensemble metrics per arm
- per_batch.csv:   accuracy / balanced accuracy of each arm within each batch (GSE149438 KRp1/KRp2, HRA003209
                   hospital): batch is aliased with class in both cohorts, so this must accompany the headline numbers
- comparisons.csv: paired patient-level bootstrap CIs / permutation p-values for the configured arm pairs
Only runs with RUN_COMPLETE.json are read; an arm with missing (seed, fold) runs is reported and excluded from the
comparisons rather than silently compared on a subset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfdna_origin.evaluation.compare import arm_probabilities, compare_arms
from cfdna_origin.evaluation.metrics import PRIMARY, all_metrics


def collect(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, preds = [], []
    for done in sorted(exp_dir.glob("*/seed_*/fold_*/RUN_COMPLETE.json")):
        d = done.parent
        m = json.loads((d / "metrics.json").read_text())
        runs.append({**{k: m["run"][k] for k in ("representation", "seed", "fold", "best_epoch", "train_seconds")},
                     **{f"test_{k}": v for k, v in m["test/all"]["primary"].items()},
                     "trainable_parameters": m["run"]["trainable_parameters"]["trainable"], "run_dir": str(d)})
        preds.append(pd.read_parquet(d / "predictions.parquet"))
    if not runs:
        raise FileNotFoundError(f"no completed runs under {exp_dir}")
    return pd.DataFrame(runs), pd.concat(preds, ignore_index=True)


def summarize(exp_dir: Path, classes: list[str], seeds: list[int], folds: list[int], pairs: list[tuple[str, str]],
              *, n_boot: int = 2000, n_perm: int = 2000, subset: str = "all") -> dict:
    runs, preds = collect(exp_dir)
    test = preds[(preds.split == "test") & (preds.eval_subset == subset)]
    expected = {(s, f) for s in seeds for f in folds}
    complete_arms, incomplete = [], {}
    for rep, g in runs.groupby("representation"):
        missing = expected - set(zip(g.seed, g.fold))
        (incomplete.__setitem__(rep, sorted(missing)) if missing else complete_arms.append(rep))
    cls_idx = {c: i for i, c in enumerate(classes)}
    cols = [f"prob_{c}" for c in classes]
    per_seed = []
    for (rep, seed), g in test.groupby(["representation", "seed"]):
        m = all_metrics(g.true_label.map(cls_idx).to_numpy(), g[cols].to_numpy(), classes)
        per_seed.append({"representation": rep, "seed": seed, "n_patients": m["n_samples"], **m["primary"],
                         **m["secondary"]})
    per_seed = pd.DataFrame(per_seed)
    per_batch = []
    if "batch" in test:
        for (rep, seed, batch), g in test.groupby(["representation", "seed", "batch"]):
            y = g.true_label.map(cls_idx).to_numpy(); p = g[cols].to_numpy().argmax(1)
            recall = [float((p[y == c] == c).mean()) for c in np.unique(y)]
            per_batch.append({"representation": rep, "seed": seed, "batch": batch, "n": len(g),
                              "classes_present": int(len(np.unique(y))), "accuracy": float((p == y).mean()),
                              "balanced_accuracy": float(np.mean(recall))})
    per_batch = pd.DataFrame(per_batch)
    arms = []
    for rep in sorted(per_seed.representation.unique()):
        ps = per_seed[per_seed.representation == rep]
        row = {"representation": rep, "complete": rep in complete_arms, "n_seeds": len(ps)}
        for k in PRIMARY:
            row[f"{k}_mean"] = ps[k].mean(); row[f"{k}_sd"] = ps[k].std(ddof=1) if len(ps) > 1 else np.nan
        if rep in complete_arms:
            y, prob, _ = arm_probabilities(test[test.representation == rep], classes)
            ens = all_metrics(y, prob, classes)["primary"]
            row.update({f"{k}_seed_ensemble": v for k, v in ens.items()})
        arms.append(row)
    arms = pd.DataFrame(arms)
    comp = compare_arms(test[test.representation.isin(complete_arms)], classes, pairs, n_boot=n_boot, n_perm=n_perm)
    out = exp_dir / "summary"
    out.mkdir(exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    per_batch.to_csv(out / "per_batch.csv", index=False)
    arms.to_csv(out / "arms.csv", index=False)
    comp.to_csv(out / "comparisons.csv", index=False)
    (out / "incomplete_arms.json").write_text(json.dumps(incomplete, indent=2, default=str))
    return {"runs": runs, "per_seed": per_seed, "per_batch": per_batch, "arms": arms, "comparisons": comp, "incomplete": incomplete}
