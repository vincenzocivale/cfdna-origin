"""One run = (dataset, experiment, representation, seed, fold).

Output directory: {outputs_root}/<dataset>/<experiment>/<representation>/seed_<seed>/fold_<fold>/ with
  config.yaml, dataset_manifest.json, representation_manifest.json, split_manifest.json, training_history.json,
  checkpoint.pt, predictions.parquet, metrics.json, environment.json, RUN_COMPLETE.json
RUN_COMPLETE.json is written last, only after: training finished, best checkpoint reloaded, test predicted once,
predictions saved and re-read, metrics validated.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cfdna_origin.config import load_paths, resolve_path
from cfdna_origin.data.fragments import FragmentStore, heldout_locus_mask
from cfdna_origin.data.schema import apply_label_scheme, validate_samples
from cfdna_origin.data.splits import LabelGuard, check_no_leakage, load_or_create_splits
from cfdna_origin.evaluation.metrics import all_metrics, softmax
from cfdna_origin.experiments.provenance import environment, now, stable_hash, write_json
from cfdna_origin.models.classifier import MILClassifier, count_parameters
from cfdna_origin.representations.registry import build_store, load_materialized, materialize, projection_tag
from cfdna_origin.training.trainer import LocusTable, predict, train

RUN_FILES = ["config.yaml", "dataset_manifest.json", "representation_manifest.json", "split_manifest.json",
             "training_history.json", "checkpoint.pt", "predictions.parquet", "metrics.json", "environment.json"]


class RunIncompleteError(RuntimeError):
    pass


def load_dataset(ds_cfg: dict, paths: dict) -> tuple[FragmentStore, pd.DataFrame, list[str]]:
    processed = resolve_path(ds_cfg["processed_dir"], paths)
    if not (processed / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"dataset {ds_cfg['name']!r} is not prepared: {processed}/dataset_manifest.json missing "
                                f"(run scripts/prepare_{ds_cfg['name']}.py)")
    fr = ds_cfg.get("fragments", {})
    store = FragmentStore(processed, min_cpgs=fr.get("min_cpgs", 3), max_cpgs=fr.get("max_cpgs", 32))
    samples = validate_samples(pd.read_parquet(processed / "samples.parquet"), sample_type=ds_cfg.get("sample_type"))
    samples, classes = apply_label_scheme(samples, ds_cfg["label_schemes"][ds_cfg["label_scheme"]])
    return store, samples, classes


def split_path(exp: dict, paths: dict) -> Path:
    ds = exp["dataset"]
    tag = stable_hash({"split": ds["split"], "label_scheme": ds["label_scheme"]})[:10]
    return paths["outputs_root"] / ds["name"] / "splits" / f"{ds['split']['strategy']}_{tag}.parquet"


def representation_table(exp: dict, rep_name: str, store: FragmentStore, paths: dict, log=print):
    rep_cfg = exp["representations"][rep_name]
    proj = exp.get("representation_projection")
    out = paths["cache_root"] / "representations" / exp["dataset"]["name"] / f"{rep_name}__{projection_tag(proj)}"
    if not (out / "manifest.json").exists():
        log(f"materialising {rep_name} on {len(store.loci)} loci -> {out}")
        materialize(build_store(rep_cfg, paths), np.asarray(store.loci), out, dataset_manifest_sha=store.manifest_sha256,
                    projection=proj, shuffle_loci_seed=rep_cfg.get("shuffle_loci_seed"))
    return load_materialized(out, len(store.loci), store.manifest_sha256)


def run_dir(exp: dict, rep: str, seed: int, fold: int, paths: dict) -> Path:
    return paths["outputs_root"] / exp["dataset"]["name"] / exp["name"] / rep / f"seed_{seed}" / f"fold_{fold}"


def is_complete(d: Path) -> bool:
    return (d / "RUN_COMPLETE.json").exists()


def _prediction_frame(ids, samples_split: pd.DataFrame, logits, classes, store, rep, seed, fold, split, subset):
    prob = softmax(logits)
    s = samples_split.set_index("sample_id").loc[ids]
    stats = pd.DataFrame([store.sample_stats(i) for i in ids])
    df = pd.DataFrame({"sample_id": ids, "patient_id": s.patient_id.to_numpy(), "true_label": s.label.to_numpy(),
                       "predicted_label": [classes[i] for i in prob.argmax(1)]})
    for j, c in enumerate(classes):
        df[f"prob_{c}"] = prob[:, j]
    df = pd.concat([df, stats], axis=1)
    if "batch" in s:
        df["batch"] = s.batch.to_numpy()
    df["representation"] = rep; df["seed"] = seed; df["fold"] = fold; df["split"] = split; df["eval_subset"] = subset
    return df


def run_one(exp: dict, rep: str, seed: int, fold: int, *, device: str | None = None, force: bool = False,
            log=print) -> Path:
    paths = load_paths() if "_paths" not in exp else exp["_paths"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(exp, rep, seed, fold, paths)
    if is_complete(out) and not force:
        log(f"skip (complete): {out}"); return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "RUN_COMPLETE.json").unlink(missing_ok=True)
    t_start = time.time()
    cfg_run = {k: v for k, v in exp.items() if not k.startswith("_")}
    cfg_run["run"] = {"representation": rep, "seed": seed, "fold": fold, "device": device}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_run, sort_keys=False))
    write_json(out / "environment.json", environment())

    store, samples, classes = load_dataset(exp["dataset"], paths)
    write_json(out / "dataset_manifest.json", {**store.manifest, "manifest_sha256": store.manifest_sha256,
                                               "classes": classes, "label_scheme": exp["dataset"]["label_scheme"]})
    splits, split_manifest = load_or_create_splits(split_path(exp, paths), samples, exp["dataset"]["split"])
    check_no_leakage(splits)
    fold_splits = splits[splits.fold == fold]
    if fold_splits.empty:
        raise ValueError(f"fold {fold} not in split table")
    guard = LabelGuard(samples, fold_splits)
    tr, va = guard.samples("train"), guard.samples("val")
    counts = {s: guard.samples(s).shape[0] for s in ("train", "val", "test")}
    write_json(out / "split_manifest.json", {**split_manifest, "fold": fold, "path": str(split_path(exp, paths)),
                                             "n_samples": counts,
                                             "train_class_counts": tr.label.value_counts().to_dict(),
                                             "val_class_counts": va.label.value_counts().to_dict()})

    table_np, rep_manifest = representation_table(exp, rep, store, paths, log)
    tcfg = exp["training"]
    table = LocusTable(table_np, device, tcfg.get("gpu_table_max_gb", 4.0))
    torch.manual_seed(seed); np.random.seed(seed)
    model = MILClassifier(table.dim, len(classes), exp["model"]).to(device)
    params = count_parameters(model)
    write_json(out / "representation_manifest.json", {**rep_manifest, "trainable_parameters": params,
                                                      "table_on_device": table.on_device})

    ul = exp.get("unseen_locus", {}) or {}
    heldout = heldout_locus_mask(np.asarray(store.loci), ul.get("fraction", 0.0), ul.get("seed", 0)) \
        if ul.get("fraction", 0) > 0 else None

    t0 = time.time()
    res = train(model, store, table, train_ids=tr.sample_id.tolist(), train_labels=guard.labels("train"),
                val_ids=va.sample_id.tolist(), val_labels=guard.labels("val"), n_classes=len(classes), cfg=tcfg,
                seed=seed, device=device, heldout=heldout, log=log)
    train_sec = time.time() - t0
    if res["best_state"] is None:
        raise RunIncompleteError("no checkpoint was selected")
    torch.save({"state_dict": res["best_state"], "best_epoch": res["best_epoch"], "classes": classes,
                "locus_dim": table.dim, "model_cfg": exp["model"]}, out / "checkpoint.pt")
    write_json(out / "training_history.json", {"history": res["history"], "best_epoch": res["best_epoch"],
                                               "best_val": res["best_val"], "train_seconds": train_sec})

    # ---- model selection is over: reload the best checkpoint, then (and only then) reveal and evaluate test
    model.load_state_dict(torch.load(out / "checkpoint.pt", weights_only=False)["state_dict"])
    guard.reveal_test()
    te = guard.samples("test")
    ev = dict(max_fragments=exp["dataset"].get("fragments", {}).get("max_eval_fragments"),
              chunk=tcfg["eval_chunk"], seed=tcfg["eval_seed"], device=device)
    t0 = time.time()
    frames = []
    for split, df in (("val", va), ("test", te)):
        logits = predict(model, store, table, df.sample_id.tolist(), **ev)
        frames.append(_prediction_frame(df.sample_id.tolist(), df, logits, classes, store, rep, seed, fold, split, "all"))
    if heldout is not None:  # unseen-locus protocol: test fragments touching loci never seen in training
        keep = [s for s in te.sample_id if len(store.valid_fragments(s, heldout, "only"))]
        logits = predict(model, store, table, keep, heldout=heldout, heldout_mode="only", **ev)
        frames.append(_prediction_frame(keep, te, logits, classes, store, rep, seed, fold, "test", "heldout_loci_only"))
    infer_sec = time.time() - t0
    pred = pd.concat(frames, ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)

    # ---- validate and summarise
    back = pd.read_parquet(out / "predictions.parquet")
    main_test = back[(back.split == "test") & (back.eval_subset == "all")]
    if set(main_test.sample_id) != set(te.sample_id) or main_test.sample_id.duplicated().any():
        raise RunIncompleteError("test predictions do not cover the test samples exactly once")
    cls_idx = {c: i for i, c in enumerate(classes)}
    prob_cols = [f"prob_{c}" for c in classes]
    metrics = {}
    for (split, subset), g in back.groupby(["split", "eval_subset"]):
        metrics[f"{split}/{subset}"] = all_metrics(g.true_label.map(cls_idx).to_numpy(), g[prob_cols].to_numpy(), classes)
    m = metrics["test/all"]["primary"]
    if not all(np.isfinite(m[k]) for k in ("macro_f1", "balanced_accuracy", "accuracy")):
        raise RunIncompleteError(f"non-finite test metrics: {m}")
    metrics["run"] = {
        "dataset": exp["dataset"]["name"], "experiment": exp["name"], "representation": rep, "seed": seed, "fold": fold,
        "n_patients": {s: int(guard.samples(s).patient_id.nunique()) for s in ("train", "val", "test")},
        "class_distribution": {s: guard.samples(s).label.value_counts().to_dict() for s in ("train", "val", "test")},
        "n_reads": int(sum(store.n_fragments(s) for s in samples.sample_id)),
        "n_valid_reads": int(sum(store.sample_stats(s)["number_of_valid_reads"] for s in samples.sample_id)),
        "n_cpg_loci": int(len(store.loci)), "trainable_parameters": params,
        "train_seconds": train_sec, "inference_seconds": infer_sec, "total_seconds": time.time() - t_start,
        "best_epoch": res["best_epoch"],
    }
    write_json(out / "metrics.json", metrics)
    missing = [f for f in RUN_FILES if not (out / f).exists()]
    if missing:
        raise RunIncompleteError(f"missing run files: {missing}")
    write_json(out / "RUN_COMPLETE.json", {"completed_utc": now(), "files": RUN_FILES,
                                           "test_primary": m, "config_hash": stable_hash(cfg_run)})
    log(f"done {out}: test macro-F1 {m['macro_f1']:.3f} bal-acc {m['balanced_accuracy']:.3f}")
    return out
