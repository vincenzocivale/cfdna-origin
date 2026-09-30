#!/usr/bin/env python3
"""Run the processed-beta pilot grid: experiments x arms x seeds x folds, in parallel over GPUs.

  python scripts/run_beta_benchmark.py --experiments beta_all_classes beta_exclude_crc --gpus 4,5 --procs-per-gpu 3
  python scripts/run_beta_benchmark.py --experiments beta_archsel_mean ... --dry-run       # print the job list

Splits and representation tables are materialised once in the parent (no races); completed runs are skipped.
An unavailable representation (e.g. ntv3_pre not materialised) is reported and skipped for that arm only.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback

from cfdna_origin.config import load_experiment, load_paths
from cfdna_origin.representations.registry import RepresentationUnavailableError


def _jobs(exp_names, arms_filter, seeds, folds):
    out = []
    for name in exp_names:
        exp = load_experiment(name)
        arms = list(exp["representations"]) + list(exp.get("baseline_arms", []))
        for arm in arms if not arms_filter else [a for a in arms if a in arms_filter]:
            for seed in seeds or exp["seeds"]:
                for fold in folds if folds is not None else exp["folds"]:
                    out.append((name, arm, seed, fold))
    return out


def _prepare(exp_names, jobs):
    from cfdna_origin.experiments.beta_runner import BASELINES, load_beta, materialize_splits
    from cfdna_origin.experiments.runner import representation_table

    paths = load_paths()
    unavailable = {}
    for name in exp_names:
        exp = load_experiment(name)
        materialize_splits(exp, sorted({s for n, _, s, _ in jobs if n == name}), paths)
        data = load_beta(exp, paths)
        for arm in sorted({a for n, a, _, _ in jobs if n == name and a not in BASELINES}):
            try:
                representation_table(exp, arm, data, paths)
            except RepresentationUnavailableError as e:
                unavailable[arm] = str(e)
    return unavailable


def _worker(queue, results, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from cfdna_origin.experiments.beta_runner import run_beta

    cache = {}
    while (job := queue.get()) is not None:
        name, arm, seed, fold = job
        try:
            exp = cache.setdefault(name, load_experiment(name))
            run_beta(exp, arm, seed, fold, device="cuda:0" if gpu != "cpu" else "cpu", log=lambda *_: None)
            results.put((job, "ok", ""))
        except RepresentationUnavailableError as e:
            results.put((job, "unavailable", str(e)))
        except Exception:
            results.put((job, "failed", traceback.format_exc()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--folds", nargs="*", type=int)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, or 'cpu'")
    ap.add_argument("--procs-per-gpu", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = _jobs(args.experiments, args.arms, args.seeds, args.folds)
    print(f"{len(jobs)} runs")
    if args.dry_run:
        for j in jobs:
            print(*j)
        return
    unavailable = _prepare(args.experiments, jobs)
    for arm, msg in unavailable.items():
        print(f"UNAVAILABLE {arm}: {msg}")
    jobs = [j for j in jobs if j[1] not in unavailable]
    ctx = mp.get_context("spawn")
    queue, results = ctx.Queue(), ctx.Queue()
    for j in jobs:
        queue.put(j)
    gpus = args.gpus.split(",")
    workers = [ctx.Process(target=_worker, args=(queue, results, g)) for g in gpus for _ in range(args.procs_per_gpu)]
    for _ in workers:
        queue.put(None)
    for w in workers:
        w.start()
    failed = []
    for i in range(len(jobs)):
        job, status, msg = results.get()
        print(f"[{i + 1}/{len(jobs)}] {status} {' '.join(map(str, job))}", flush=True)
        if status == "failed":
            failed.append(job); print(msg, flush=True)
    for w in workers:
        w.join()
    if failed or unavailable:
        print(f"failed: {failed}\nunavailable arms (skipped, others unaffected): {sorted(unavailable)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
