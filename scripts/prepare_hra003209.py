#!/usr/bin/env python3
"""HRA003209 (MONITOR) preparation. Controlled access: implementable and checkable before the data arrive.

Steps:
  metadata   build <meta_dir>/samples_meta.tsv from the paper's Supplementary Data 4 (official Training/Test split,
             hospital, stage) and the GSA public metadata (patient -> BAM run). Both files are public; place them in
             <meta_dir> (download URLs printed if missing). The paper's model scores are dropped.
  validate   list exactly which inputs are missing (BAMs, chain file, hg38 FASTA); exit code 1 if anything is missing
  process    per sample: Bismark hg19 BAM -> fragments -> liftover to GRCh38 (drop + count loci that are unmapped,
             ambiguous, or not CpGs in hg38; fragments reduced below min CpGs are kept here and filtered at load time)
  finalize   loci union, samples.parquet, dataset_manifest.json
No sample is ever fabricated: every step fails listing what is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cfdna_origin.config import load_component, load_paths, resolve_path
from cfdna_origin.data.bam_fragments import extract_fragments
from cfdna_origin.data.datasets import hra003209 as ds
from cfdna_origin.data.fragments import filter_calls, finalize_dataset, write_sample
from cfdna_origin.data.liftover import Chain
from cfdna_origin.data.reference import reference_cpg_keys
from cfdna_origin.data.schema import validate_samples

CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"


def _cfg():
    paths = load_paths()
    cfg = load_component("datasets", "hra003209")
    for k in ("meta_dir", "raw_dir", "processed_dir", "liftover_chain"):
        cfg[k] = resolve_path(cfg[k], paths)
    cfg["hg38_fasta"] = paths["reference_root"] / "hg38" / "hg38.fa.gz"
    cfg["hg38_cpg_cache"] = paths["reference_root"] / "hg38" / "cpg_keys_autosomes.npy"
    return cfg


def cmd_metadata(cfg, args) -> None:
    supp, gsa = cfg["meta_dir"] / "paper_supp" / "MOESM5_ESM.xlsx", cfg["meta_dir"] / "HRA003209_public_metadata.xlsx"
    missing = [(p, u) for p, u in ((supp, ds.SUPP_URL), (gsa, ds.GSA_BROWSE_URL + " (Download metadata)")) if not p.exists()]
    if missing:
        raise SystemExit("missing public metadata:\n" + "\n".join(f"  {p}  <- {u}" for p, u in missing))
    table = ds.build_sample_table(supp, gsa)
    table.to_csv(cfg["meta_dir"] / "samples_meta.tsv", sep="\t", index=False)
    print(pd.crosstab(table.diagnosis, table.official_split, margins=True))
    print(f"-> {cfg['meta_dir'] / 'samples_meta.tsv'}")


def _missing(cfg) -> list[str]:
    meta_file = cfg["meta_dir"] / "samples_meta.tsv"
    if not meta_file.exists():
        return [f"{meta_file} (run the 'metadata' step)"]
    meta = pd.read_csv(meta_file, sep="\t")
    out = ds.missing_inputs(meta, cfg["raw_dir"])
    for p, hint in ((cfg["liftover_chain"], CHAIN_URL), (cfg["hg38_fasta"], "scripts/prepare_gse149438.py reference")):
        if not p.exists():
            out.append(f"{p}  <- {hint}")
    return out


def cmd_validate(cfg, args) -> None:
    missing = _missing(cfg)
    if missing:
        bams = [m for m in missing if m.endswith(".bam")]
        other = [m for m in missing if not m.endswith(".bam")]
        print(f"MISSING: {len(bams)} BAM files (expected under {cfg['raw_dir']}), e.g. {bams[:3]}")
        for m in other:
            print("MISSING:", m)
        sys.exit(1)
    print("all inputs present")


def _process_one(row: dict, cfg: dict) -> dict:
    row = pd.Series(row)
    out = cfg["processed_dir"] / "fragments" / row.sample_id
    if (out / "extract_stats.json").exists():
        return {"sample_id": row.sample_id, "status": "done"}
    pp = cfg["preprocessing"]
    hg19_contigs = cfg["contigs"] + [c[3:] for c in cfg["contigs"]]  # Bismark hg19 BAMs may use "1" or "chr1"
    off, keys, state, stats = extract_fragments(ds.expected_bam(cfg["raw_dir"], row), contigs=hg19_contigs,
                                                min_mapq=pp["min_mapq"], require_proper_pair=pp["require_proper_pair"])
    chain = Chain(cfg["liftover_chain"])
    lifted, ok, lstats = chain.lift(keys, np.load(cfg["hg38_cpg_cache"], mmap_mode="r"))
    off, keys, state, fstats = filter_calls(off, lifted, state, ok)
    frag_of = np.repeat(np.arange(len(off) - 1), np.diff(off))
    order = np.lexsort((keys, frag_of))  # re-sort CpGs by position within each fragment (minus-strand chains)
    keys, state = keys[order], state[order]
    res = write_sample(out, off, keys, state)
    stats.update({f"liftover_{k}": v for k, v in lstats.items()}, **res,
                 fragments_emptied_by_liftover=fstats["fragments_emptied"], finished_utc=datetime.now(timezone.utc).isoformat())
    (out / "extract_stats.json").write_text(json.dumps(stats, indent=2))
    return {"sample_id": row.sample_id, "status": "ok", **res}


def cmd_process(cfg, args) -> None:
    missing = _missing(cfg)
    meta = pd.read_csv(cfg["meta_dir"] / "samples_meta.tsv", sep="\t")
    if args.samples:
        meta = meta[meta.sample_id.isin(args.samples)]
    need = {str(ds.expected_bam(cfg["raw_dir"], r)) for r in meta.itertuples()}
    blocking = [m for m in missing if not m.endswith(".bam") or m in need]
    if blocking:
        raise SystemExit(f"cannot process: {len(blocking)} missing inputs, e.g. {blocking[:5]} (run 'validate')")
    reference_cpg_keys(cfg["hg38_fasta"], cfg["hg38_cpg_cache"], cfg["contigs"])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_process_one, r, cfg) for r in meta.to_dict("records")]
        for f in as_completed(futs):
            print(json.dumps(f.result()), flush=True)


def cmd_finalize(cfg, args) -> None:
    meta = pd.read_csv(cfg["meta_dir"] / "samples_meta.tsv", sep="\t")
    frag = cfg["processed_dir"] / "fragments"
    done = [s for s in meta.sample_id if (frag / s / "extract_stats.json").exists()]
    missing = sorted(set(meta.sample_id) - set(done))
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} samples not processed (e.g. {missing[:5]})")
    stats = {s: json.loads((frag / s / "extract_stats.json").read_text()) for s in done}
    table = meta[meta.sample_id.isin(done)].copy()
    table["n_fragments"] = table.sample_id.map(lambda s: stats[s]["n_fragments"])
    table["n_cpgs"] = table.sample_id.map(lambda s: stats[s]["n_cpgs"])
    table = validate_samples(table)
    build = {"dataset": "hra003209", "genome_build": cfg["genome_build"], "source_genome_build": cfg["source_genome_build"],
             "contigs": cfg["contigs"], "preprocessing": cfg["preprocessing"], "partial": bool(missing),
             "missing_samples": missing, "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest = finalize_dataset(cfg["processed_dir"], table.sample_id.tolist(), build)
    table.to_parquet(cfg["processed_dir"] / "samples.parquet", index=False)
    print(json.dumps({k: manifest[k] for k in ("n_samples", "n_loci", "partial")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("metadata"); sub.add_parser("validate")
    p = sub.add_parser("process"); p.add_argument("--samples", nargs="*"); p.add_argument("--workers", type=int, default=8)
    f = sub.add_parser("finalize"); f.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    cfg = _cfg()
    {"metadata": cmd_metadata, "validate": cmd_validate, "process": cmd_process, "finalize": cmd_finalize}[args.step](cfg, args)


if __name__ == "__main__":
    main()
