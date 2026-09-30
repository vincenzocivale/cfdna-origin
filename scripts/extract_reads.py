#!/usr/bin/env python3
"""Extract a thinned, labelled read table per sample from the hg38 PAT files (resumable, parallel)."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cfdna_too.data.cpg_index import CpGIndex
from cfdna_too.data.reads import extract_sample

ROOT = Path(__file__).resolve().parents[1]


def work(args):
    gsm, fname, size, a = args
    out = a["out"] / f"{gsm}.npz"
    if out.exists() and out.stat().st_size > 0:
        return gsm, "exists", None
    pat = a["pat_dir"] / fname
    if not pat.exists() or pat.stat().st_size != size:
        return gsm, "incomplete_download", None
    arrays, stats = extract_sample(pat, CpGIndex(ROOT / "data/hg38_cpg_index.npz"), p=a["p"], min_cpg=a["min_cpg"],
                                   max_cpg=a["max_cpg"], seed_key=gsm)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(out)
    (a["out"] / f"{gsm}.json").write_text(json.dumps(stats))
    return gsm, "done", stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, default=ROOT / "data/meta/samples.parquet")
    ap.add_argument("--pat-dir", type=Path, default=ROOT.parent / "CpGRepresentationBenchmark/data/external/wgbs_atlas/hg38_pat")
    ap.add_argument("--out", type=Path, default=ROOT / "data/reads")
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--min-cpg", type=int, default=4)
    ap.add_argument("--max-cpg", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", help="restrict to these GSM ids")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    listing = pd.read_csv(args.pat_dir.parent / "filelist.txt", sep="\t")
    sizes = dict(zip(listing.Name, listing.Size))
    df = pd.read_parquet(args.samples)
    if args.only:
        df = df[df.gsm.isin(args.only)]
    cfg = dict(out=args.out, pat_dir=args.pat_dir, p=args.p, min_cpg=args.min_cpg, max_cpg=args.max_cpg)
    jobs = [(r.gsm, r.pat_file, int(sizes[r.pat_file]), cfg) for r in df.itertuples()]
    with ProcessPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(work, j) for j in jobs]):
            gsm, status, st = f.result()
            print(gsm, status, st["kept_reads"] if st else "", flush=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))


if __name__ == "__main__":
    main()
