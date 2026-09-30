#!/usr/bin/env python3
"""Apply the pre-registered selection rule (docs/BETA_ARCHITECTURE_SELECTION.md) to the validation-only runs.

Reads metrics.json["val/all"] of beta_archsel_{mean,deepsets,pma}; writes configs/models/cpgset_selected.yaml and the
result table into the doc. Refuses to run if any selection run is missing or if a run contains test predictions.
"""
from __future__ import annotations

import json

import pandas as pd
import yaml

from cfdna_origin.config import CONFIG_DIR, REPO_ROOT, load_component, load_experiment, load_paths

CANDIDATES = ["mean", "deepsets", "pma"]  # simplicity order for the tie rule
TIE = 0.01


def main() -> None:
    paths = load_paths()
    rows = []
    for arch in CANDIDATES:
        exp = load_experiment(f"beta_archsel_{arch}")
        root = paths["outputs_root"] / exp["dataset"]["name"] / exp["name"]
        for arm in exp["representations"]:
            for seed in exp["seeds"]:
                for fold in exp["folds"]:
                    d = root / arm / f"seed_{seed}" / f"fold_{fold}"
                    if not (d / "RUN_COMPLETE.json").exists():
                        raise SystemExit(f"missing selection run {d}")
                    m = json.loads((d / "metrics.json").read_text())
                    if any(k.startswith("test/") for k in m):
                        raise SystemExit(f"{d} contains test metrics: selection runs must be validation-only")
                    rows.append({"arch": arch, "arm": arm, "fold": fold, **m["val/all"]["primary"],
                                 "params": m["run"]["trainable_parameters"]["trainable"],
                                 "best_epoch": m["run"]["best_epoch"]})
    df = pd.DataFrame(rows)
    table = df.groupby("arch").agg(val_macro_f1=("macro_f1", "mean"), val_macro_f1_sd=("macro_f1", "std"),
                                   val_bal_acc=("balanced_accuracy", "mean"), val_auroc=("auroc_macro", "mean"),
                                   params=("params", "first"), mean_best_epoch=("best_epoch", "mean")).loc[CANDIDATES]
    per_arm = df.pivot_table(index="arch", columns="arm", values="macro_f1", aggfunc="mean").loc[CANDIDATES]
    best = table.val_macro_f1.max()
    chosen = next(a for a in CANDIDATES if table.loc[a, "val_macro_f1"] >= best - TIE)
    cfg = load_component("models", f"cpgset_{chosen}")
    cfg["name"] = "cpgset_selected"
    cfg["selected_from"] = f"cpgset_{chosen}"
    (CONFIG_DIR / "models" / "cpgset_selected.yaml").write_text(
        "# Written by scripts/select_beta_architecture.py (docs/BETA_ARCHITECTURE_SELECTION.md). Frozen for all arms.\n"
        + yaml.safe_dump(cfg, sort_keys=False))
    doc = REPO_ROOT / "docs" / "BETA_ARCHITECTURE_SELECTION.md"
    text = doc.read_text().split("## Result")[0]
    text += ("## Result\n\nValidation only (5 folds x 2 representation-neutral arms, seed 17):\n\n"
             + table.round(4).to_markdown() + "\n\nValidation macro-F1 per development arm:\n\n"
             + per_arm.round(4).to_markdown()
             + f"\n\n**Selected: `{chosen}`** (best mean {best:.4f}; tie margin {TIE}; simplicity order "
             + " < ".join(CANDIDATES) + "). Frozen as `configs/models/cpgset_selected.yaml`.\n")
    doc.write_text(text)
    print(table); print(per_arm); print("selected:", chosen)


if __name__ == "__main__":
    main()
