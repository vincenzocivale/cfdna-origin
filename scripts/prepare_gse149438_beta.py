#!/usr/bin/env python3
"""Build the `gse149438_processed_beta` dataset from the public GEO methratio tables (hg19 -> GRCh38).

  python scripts/prepare_gse149438.py metadata          # once: samples_meta.tsv (labels, batch, patient ids)
  python scripts/prepare_gse149438_beta.py [--workers 16] [--max-gb 10]

Inputs (fail loudly if missing): GSE149438_RAW.tar (GEO series supplementary; streamed, never extracted),
hg19ToHg38.over.chain.gz (UCSC), GRCh38 FASTA (for the CpG check), the functional representation store.
Outputs in <processed_dir>: loci.npy (GRCh38 keys), loci_hg19.npy (aligned source keys), beta.npy float16
[samples x loci] (NaN = not observed), coverage.npy uint16 (0 = not observed), samples.parquet (+ per-sample summary
statistics), liftover_manifest.json, dataset_manifest.json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cfdna_origin.config import load_component, load_paths, resolve_path
from cfdna_origin.data.datasets import gse149438_beta as ds
from cfdna_origin.data.fragments import sha256_array
from cfdna_origin.data.liftover import Chain
from cfdna_origin.data.reference import reference_cpg_keys
from cfdna_origin.representations.registry import build_store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-gb", type=float, default=10.0, help="refuse to write matrices larger than this")
    args = ap.parse_args()
    paths = load_paths()
    cfg = load_component("datasets", "gse149438_processed_beta")
    p = {k: resolve_path(cfg[k], paths) for k in ("meta_dir", "geo_tar", "processed_dir", "liftover_chain",
                                                   "hg38_fasta", "hg38_cpg_cache")}
    missing = [f"{k}: {v}" for k, v in p.items() if k not in ("processed_dir", "hg38_cpg_cache") and not v.exists()]
    if not (p["meta_dir"] / "samples_meta.tsv").exists():
        missing.append(f"{p['meta_dir'] / 'samples_meta.tsv'} (run prepare_gse149438.py metadata)")
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))

    meta = pd.read_csv(p["meta_dir"] / "samples_meta.tsv", sep="\t")
    members = ds.list_members(p["geo_tar"])
    samples = ds.build_samples_table(meta, members).sort_values("sample_id").reset_index(drop=True)
    print(f"parsing {len(samples)} methratio tables from {p['geo_tar'].name}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        parsed = list(pool.map(ds.read_member, [p["geo_tar"]] * len(samples), samples.source_file,
                               [cfg["min_coverage"]] * len(samples)))
    parse_stats = {sid: st for sid, (_, _, _, st) in zip(samples.sample_id, parsed)}

    # ---- liftover of the union of observed hg19 loci
    union = np.unique(np.concatenate([k for k, _, _, _ in parsed]))
    ref = reference_cpg_keys(p["hg38_fasta"], p["hg38_cpg_cache"], cfg["contigs"])
    chain = Chain(p["liftover_chain"])
    target, ok_nocheck, st_nocheck = chain.lift(union)
    _, ok, st = chain.lift(union, ref)
    ok, n_collision = ds.resolve_collisions(target, ok)
    stores = {r: build_store(load_component("representations", r), paths) for r in cfg["required_representations"]}
    in_stores, found = ok.copy(), {}
    for name, store in stores.items():
        cov_ok = store.covers(target[ok])
        found[name] = int(cov_ok.sum())
        in_stores[ok] &= cov_ok
    manifest = {
        "n_input_loci": int(len(union)),
        "n_lifted": int(ok_nocheck.sum()),
        "n_failed": st_nocheck["unmapped"],
        "n_ambiguous": st_nocheck["ambiguous"],
        "n_non_CpG_hg38": st["not_cpg_in_target"],
        "n_many_to_one_collisions": n_collision,
        "n_valid_hg38_cpg": int(ok.sum()),
        "n_found_in_representation_store": found,
        "n_final_loci": int(in_stores.sum()),
        "chain": str(p["liftover_chain"]), "reference": str(p["hg38_fasta"]),
        "rules": "unmapped/ambiguous dropped; target must be a GRCh38 CpG C (1-based, + strand); many-to-one dropped; "
                 "target must be in every required representation store",
    }
    print(json.dumps(manifest, indent=2), flush=True)
    order = np.argsort(target[in_stores], kind="stable")
    loci = target[in_stores][order]
    loci_hg19 = union[in_stores][order]
    col_of_union = np.full(len(union), -1, np.int64)
    col_of_union[np.flatnonzero(in_stores)[order]] = np.arange(len(loci))

    est_gb = len(samples) * len(loci) * 4 / 1e9
    print(f"matrices: {len(samples)} x {len(loci)} -> {est_gb:.2f} GB (beta f16 + coverage u16)", flush=True)
    if est_gb > args.max_gb:
        raise SystemExit(f"estimated {est_gb:.1f} GB > --max-gb {args.max_gb}")
    out = p["processed_dir"]; out.mkdir(parents=True, exist_ok=True)
    beta = np.lib.format.open_memmap(out / "beta.tmp.npy", mode="w+", dtype=np.float16, shape=(len(samples), len(loci)))
    cov = np.lib.format.open_memmap(out / "coverage.tmp.npy", mode="w+", dtype=np.uint16, shape=(len(samples), len(loci)))
    rows, clipped = [], 0
    for i, (sid, (keys, c, ct, _)) in enumerate(zip(samples.sample_id, parsed)):
        col = col_of_union[np.searchsorted(union, keys)]
        keep = col >= 0
        brow = np.full(len(loci), np.nan, np.float32); crow = np.zeros(len(loci), np.int64)
        brow[col[keep]] = c[keep] / ct[keep]
        crow[col[keep]] = ct[keep]
        clipped += int((crow > 65535).sum())
        beta[i] = brow.astype(np.float16); cov[i] = np.minimum(crow, 65535).astype(np.uint16)
        rows.append({"sample_id": sid, "n_input_cpgs": int(len(keys)),
                     "n_dropped_by_liftover": int((~keep).sum()), **ds.sample_summary(brow, crow)})
    beta.flush(); cov.flush(); del beta, cov
    (out / "beta.tmp.npy").replace(out / "beta.npy"); (out / "coverage.tmp.npy").replace(out / "coverage.npy")
    np.save(out / "loci.npy", loci); np.save(out / "loci_hg19.npy", loci_hg19)
    table = samples.merge(pd.DataFrame(rows), on="sample_id")
    table["n_fragments"] = 0; table["n_cpgs"] = table.n_observed_loci  # schema compatibility (no reads here)
    table.to_parquet(out / "samples.parquet", index=False)
    manifest["coverage_clipped_to_65535"] = clipped
    (out / "liftover_manifest.json").write_text(json.dumps(manifest, indent=2))
    dataset_manifest = {
        "dataset": "gse149438_processed_beta", "mode": "processed_beta", "genome_build": "GRCh38",
        "source_genome_build": "hg19", "source": str(p["geo_tar"]), "min_coverage": cfg["min_coverage"],
        "contigs": cfg["contigs"], "n_samples": int(len(table)), "n_loci": int(len(loci)),
        "loci_sha256": sha256_array(loci), "liftover": manifest, "parse_stats": parse_stats,
        "missing_policy": "not imputed: NaN beta / 0 coverage; models consume observed tokens only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2))
    print(table.groupby(["diagnosis", "batch"]).n_observed_loci.describe()[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
