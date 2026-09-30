"""Processed-beta pilot: one run = (experiment, arm, seed, fold). Same run files and completion rule as the fragment
benchmark (`experiments/runner.py`); RUN_COMPLETE.json is written last.

An experiment fixes a protocol and a task:
  protocol  all_classes_original | exclude_crc | batch_robust_subset (classes with >= min_per_batch patients in every
            batch, evaluated cross-batch: train on one batch, test on the other)
  task      tissue (sample -> class) | batch (sample -> KRp1/KRp2; confounding diagnostic only)
Arms: representation arms (CpG-set model, frozen locus table) and embedding-free baselines (summary_only,
dmr_logistic, dmr_xgboost). Repeated CV: repetition r uses split seed = training seed (every arm shares the splits).
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cfdna_origin.config import load_paths, resolve_path
from cfdna_origin.data.beta import BetaData
from cfdna_origin.data.loci import key_chrom_code, key_position
from cfdna_origin.data.schema import apply_label_scheme
from cfdna_origin.data.splits import LabelGuard, check_no_leakage, load_or_create_splits
from cfdna_origin.evaluation.metrics import all_metrics, softmax
from cfdna_origin.experiments.provenance import environment, now, stable_hash, write_json
from cfdna_origin.experiments.runner import RUN_FILES, RunIncompleteError, representation_table
from cfdna_origin.models.classical import DMRClassifier, SummaryClassifier, candidate_regions, region_matrix
from cfdna_origin.models.cpgset import CpGSetClassifier
from cfdna_origin.training import beta_trainer
from cfdna_origin.training.trainer import LocusTable

BASELINES = ("summary_only", "dmr_logistic", "dmr_xgboost")
_DATA_CACHE: dict[str, BetaData] = {}


def load_beta(exp: dict, paths: dict) -> BetaData:
    d = str(resolve_path(exp["dataset"]["processed_dir"], paths))
    if d not in _DATA_CACHE:
        _DATA_CACHE[d] = BetaData(Path(d))
    return _DATA_CACHE[d]


def task_samples(exp: dict, data: BetaData) -> tuple[pd.DataFrame, list[str]]:
    """Sample table with `label`/`label_idx` for the experiment's protocol and task; classes in fixed order."""
    ds, proto, task = exp["dataset"], exp["protocol"], exp.get("task", "tissue")
    df, classes = apply_label_scheme(data.samples, ds["label_schemes"][ds["label_scheme"]])
    df = df[~df.label.isin(proto.get("exclude_classes", []))]
    if proto["name"] == "batch_robust_subset":
        per = df.groupby(["label", "batch"]).patient_id.nunique().unstack(fill_value=0)
        keep = per.index[(per >= proto.get("min_per_batch", 8)).all(1)]
        df = df[df.label.isin(keep)]
    classes = [c for c in classes if c in set(df.label)]
    if task == "batch":
        df = df.assign(tissue_label=df.label, label=df.batch)
        classes = sorted(df.batch.unique())
    df = df.assign(label_idx=df.label.map({c: i for i, c in enumerate(classes)})).reset_index(drop=True)
    return df, classes


def split_spec(exp: dict, seed: int) -> dict:
    spec = dict(exp["dataset"]["split"])
    spec.update(exp["protocol"].get("split", {}))
    if spec.pop("seed_from_run", False):
        spec["seed"] = seed
    return spec


def split_file(exp: dict, samples: pd.DataFrame, spec: dict, paths: dict) -> Path:
    tag = stable_hash({"spec": spec, "samples": sorted(samples.sample_id), "labels": samples.label.tolist(),
                       "task": exp.get("task", "tissue")})[:10]
    return paths["outputs_root"] / exp["dataset"]["name"] / "splits" / f"{exp['protocol']['name']}_{exp.get('task', 'tissue')}_s{spec['seed']}_{tag}.parquet"


def materialize_splits(exp: dict, seeds: list[int], paths: dict | None = None) -> list[dict]:
    paths = paths or load_paths()
    samples, _ = task_samples(exp, load_beta(exp, paths))
    return [load_or_create_splits(split_file(exp, samples, split_spec(exp, s), paths), samples, split_spec(exp, s))[1]
            for s in seeds]


def _frame(df: pd.DataFrame, prob: np.ndarray, classes, arm, seed, fold, split) -> pd.DataFrame:
    out = pd.DataFrame({"sample_id": df.sample_id.to_numpy(), "patient_id": df.patient_id.to_numpy(),
                        "true_label": df.label.to_numpy(), "predicted_label": [classes[i] for i in prob.argmax(1)]})
    for j, c in enumerate(classes):
        out[f"prob_{c}"] = prob[:, j]
    for c in ("batch", "n_observed_loci", "missing_fraction", "mean_coverage", "median_coverage", "mean_beta",
              "beta_variance"):
        out[c] = df[c].to_numpy()
    if "tissue_label" in df:
        out["tissue_label"] = df.tissue_label.to_numpy()
    out["representation"] = arm; out["seed"] = seed; out["fold"] = fold; out["split"] = split; out["eval_subset"] = "all"
    return out


def _attributions(model, data, table, df, prob, classes, cfg, device, rep_table) -> pd.DataFrame:
    """Top-k CpGs per test sample by the token-alone logit of the predicted class (interpretation only)."""
    rows = []
    for (_, r), p in zip(df.iterrows(), prob):
        cols = data.observed(r.sample_id)
        pool, solo = [], []
        for s in range(0, len(cols), cfg["eval_chunk"]):
            emb, b, c, _ = beta_trainer._tensors(data, table, [r.sample_id], [cols[s : s + cfg["eval_chunk"]]], device)
            ps, sl = model.token_scores(emb, b, c)
            pool.append(ps.cpu().numpy()); solo.append(sl.float().cpu().numpy())
        pool, solo = np.concatenate(pool), np.concatenate(solo)
        pred = int(p.argmax()); true = classes.index(r.label)
        contrib = solo[:, pred] - solo[:, pred].mean()
        top = np.argsort(-contrib)[: cfg.get("attribution_top_k", 200)]
        _, beta, cov = data.tokens(r.sample_id, cols[top])
        keys = data.loci[cols[top]]
        frame = pd.DataFrame({"sample_id": r.sample_id, "true_class": r.label, "predicted_class": classes[pred],
                              "chrom_code": key_chrom_code(keys), "pos_hg38": key_position(keys),
                              "pos_hg19": key_position(data.loci_hg19[cols[top]]), "beta": beta, "coverage": cov,
                              "pool_score": pool[top], "contribution_pred": contrib[top],
                              "contribution_true": solo[top, true] - solo[:, true].mean()})
        for j in range(min(4, rep_table.shape[1])):
            frame[f"embedding_pc{j + 1}"] = np.asarray(rep_table[cols[top], j], np.float32)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run_beta(exp: dict, arm: str, seed: int, fold: int, *, device: str | None = None, force: bool = False,
             log=print) -> Path:
    paths = exp.get("_paths") or load_paths()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = paths["outputs_root"] / exp["dataset"]["name"] / exp["name"] / arm / f"seed_{seed}" / f"fold_{fold}"
    if (out / "RUN_COMPLETE.json").exists() and not force:
        log(f"skip (complete): {out}"); return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "RUN_COMPLETE.json").unlink(missing_ok=True)
    t_start = time.time()
    cfg_run = {k: v for k, v in exp.items() if not k.startswith("_")}
    cfg_run["run"] = {"arm": arm, "seed": seed, "fold": fold, "device": device}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_run, sort_keys=False))
    write_json(out / "environment.json", environment())

    data = load_beta(exp, paths)
    samples, classes = task_samples(exp, data)
    write_json(out / "dataset_manifest.json", {**{k: v for k, v in data.manifest.items() if k != "parse_stats"},
                                               "manifest_sha256": data.manifest_sha256, "classes": classes,
                                               "protocol": exp["protocol"], "task": exp.get("task", "tissue"),
                                               "n_task_samples": int(len(samples))})
    spec = split_spec(exp, seed)
    splits, split_manifest = load_or_create_splits(split_file(exp, samples, spec, paths), samples, spec)
    check_no_leakage(splits)
    guard = LabelGuard(samples, splits[splits.fold == fold])
    tr, va = guard.samples("train"), guard.samples("val")
    write_json(out / "split_manifest.json", {**split_manifest, "fold": fold,
                                             "n_samples": {s: int(len(guard.samples(s))) for s in ("train", "val", "test")}})
    tcfg = exp["training"]
    t0 = time.time()
    history, best_epoch, params, extra = {}, None, {"trainable": 0}, {}
    if arm in BASELINES:
        fit_df = pd.concat([tr, va], ignore_index=True)  # fixed hyper-parameters: no model selection -> use train+val
        y_fit = np.concatenate([guard.labels("train"), guard.labels("val")])
        if arm == "summary_only":
            model = SummaryClassifier(seed)
            info = model.fit(data.summary_features(fit_df.sample_id.tolist()), y_fit)
        else:
            bcfg = exp["baselines"][arm]
            region_of = candidate_regions(data.loci, bcfg["max_gap"])
            Xf, sizes = region_matrix(data, fit_df.sample_id.tolist(), region_of)
            model = DMRClassifier(arm, len(classes), bcfg, seed)
            info = model.fit(Xf, y_fit, sizes)
        write_json(out / "representation_manifest.json", {"name": arm, "kind": "embedding_free_baseline", **info})
        with open(out / "checkpoint.pt", "wb") as fh:
            pickle.dump(model, fh)
        history = {"history": [], "note": "fixed hyper-parameters, fit on train+val", "fit_info": info}
        extra = {"fit_info": info}
    else:
        table_np, rep_manifest = representation_table(exp, arm, data, paths, log)
        table = LocusTable(table_np, device, tcfg.get("gpu_table_max_gb", 4.0))
        torch.manual_seed(seed); np.random.seed(seed)
        model = CpGSetClassifier(table.dim, len(classes), exp["model"]).to(device)
        params = {"trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                  "representation_adapter": int(sum(p.numel() for p in model.locus.parameters()))}
        proj = rep_manifest.get("projection", {})
        write_json(out / "representation_manifest.json", {
            **rep_manifest, "raw_dim": rep_manifest.get("dim"), "projected_dim": table.dim,
            "explained_variance_ratio": proj.get("explained_variance_ratio"), "fit_universe_size": proj.get("fit_loci"),
            "pca_hash": proj.get("components_sha256"),
            "source_representation_hash": rep_manifest.get("provenance", {}).get("store_fingerprint_sha256"),
            "trainable_parameters": params})
        res = beta_trainer.train(model, data, table, train_ids=tr.sample_id.tolist(), train_labels=guard.labels("train"),
                                 val_ids=va.sample_id.tolist(), val_labels=guard.labels("val"), n_classes=len(classes),
                                 cfg=tcfg, seed=seed, device=device, log=log)
        if res["best_state"] is None:
            raise RunIncompleteError("no checkpoint selected")
        torch.save({"state_dict": res["best_state"], "best_epoch": res["best_epoch"], "classes": classes}, out / "checkpoint.pt")
        history = {"history": res["history"], "best_epoch": res["best_epoch"], "best_val": res["best_val"]}
        best_epoch = res["best_epoch"]
        model.load_state_dict(torch.load(out / "checkpoint.pt", weights_only=False)["state_dict"])
    train_sec = time.time() - t0
    write_json(out / "training_history.json", {**history, "train_seconds": train_sec})

    # ---- model fixed: reveal and evaluate test once
    if not exp.get("selection_only", False):
        guard.reveal_test()
    te = guard.samples("test")
    t0 = time.time()
    frames = []
    selection_only = bool(exp.get("selection_only", False))  # architecture selection: test is never predicted
    for split, df in (("val", va),) + ((() if selection_only else (("test", te),))):
        ids = df.sample_id.tolist()
        if arm == "summary_only":
            prob = model.predict_proba(data.summary_features(ids))
        elif arm in BASELINES:
            prob = model.predict_proba(region_matrix(data, ids, region_of)[0])
        else:
            prob = softmax(beta_trainer.predict(model, data, table, ids, max_tokens=None, chunk=tcfg["eval_chunk"],
                                                seed=tcfg["eval_seed"], device=device))
        frames.append(_frame(df, prob, classes, arm, seed, fold, split))
        if split == "test" and arm in exp.get("attribution", {}).get("arms", []) and \
                seed == exp["attribution"].get("seed", seed):
            acfg = {**tcfg, **exp["attribution"]}  # attribution_top_k lives in the `attribution` block
            _attributions(model, data, table, df, prob, classes, acfg, device, table_np).to_parquet(
                out / "attributions.parquet", index=False)
    infer_sec = time.time() - t0
    pred = pd.concat(frames, ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    back = pd.read_parquet(out / "predictions.parquet")
    test = back[back.split == "test"]
    if not selection_only and (set(test.sample_id) != set(te.sample_id) or test.sample_id.duplicated().any()):
        raise RunIncompleteError("test predictions do not cover the test samples exactly once")
    idx = {c: i for i, c in enumerate(classes)}
    cols = [f"prob_{c}" for c in classes]
    metrics = {f"{s}/all": all_metrics(g.true_label.map(idx).to_numpy(), g[cols].to_numpy(), classes)
               for s, g in back.groupby("split")}
    m = metrics["val/all" if selection_only else "test/all"]["primary"]
    if not all(np.isfinite(m[k]) for k in ("macro_f1", "balanced_accuracy", "accuracy")):
        raise RunIncompleteError(f"non-finite test metrics {m}")
    metrics["run"] = {"dataset": exp["dataset"]["name"], "experiment": exp["name"], "representation": arm, "seed": seed,
                      "fold": fold, "protocol": exp["protocol"]["name"], "task": exp.get("task", "tissue"),
                      "n_patients": {s: int(guard.samples(s).patient_id.nunique()) for s in ("train", "val", "test")},
                      "class_distribution": {s: guard.samples(s).label.value_counts().to_dict()
                                             for s in ("train", "val") + (() if selection_only else ("test",))},
                      "n_cpg_loci": int(len(data.loci)), "trainable_parameters": params, "best_epoch": best_epoch,
                      "train_seconds": train_sec, "inference_seconds": infer_sec,
                      "total_seconds": time.time() - t_start, **extra}
    write_json(out / "metrics.json", metrics)
    missing = [f for f in RUN_FILES if not (out / f).exists()]
    if missing:
        raise RunIncompleteError(f"missing run files {missing}")
    write_json(out / "RUN_COMPLETE.json", {"completed_utc": now(), "files": RUN_FILES, "test_primary": m,
                                           "config_hash": stable_hash(cfg_run)})
    log(f"done {out}: macro-F1 {m['macro_f1']:.3f} bal-acc {m['balanced_accuracy']:.3f} auroc {m['auroc_macro']:.3f}")
    return out
