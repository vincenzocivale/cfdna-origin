#!/usr/bin/env python3
"""Reproducible GSE149438 preparation from public accessions (GEO GSE149438, ENA PRJNA628686).

Steps (each resumable; run in order):
  metadata   download the GEO SOFT + ENA run report, write <meta_dir>/samples_meta.tsv (no data-derived fields)
  reference  download GRCh38 (UCSC hg38 analysis set, no alt contigs) and build the Bismark index        [needs bismark, bowtie2]
  process    per sample: ENA FASTQ -> trim_galore -> Bismark -> dedup -> fragments; large intermediates are deleted
             as soon as a sample's fragments are written (the whole FASTQ set, 224 GB, never sits on disk at once)
                                                                                    [needs aria2c, trim_galore, bismark]
  finalize   union of observed loci, per-sample locus rows, samples.parquet, dataset_manifest.json
  status     what is done / missing

Example:
  python scripts/prepare_gse149438.py metadata
  python scripts/prepare_gse149438.py reference --threads 32
  python scripts/prepare_gse149438.py process --workers 6 --threads 16 [--samples GSM4502064 ...] [--dry-run]
  python scripts/prepare_gse149438.py finalize
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cfdna_origin.config import load_component, load_paths, resolve_path
from cfdna_origin.data.bam_fragments import extract_fragments
from cfdna_origin.data.datasets import gse149438 as ds
from cfdna_origin.data.fragments import filter_calls, finalize_dataset, write_sample
from cfdna_origin.data.reference import reference_cpg_keys
from cfdna_origin.data.schema import validate_samples

HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/analysisSet/hg38.analysisSet.fa.gz"  # no alt contigs
REQUIRED_TOOLS = {"reference": ["bismark_genome_preparation", "bowtie2"],
                  "process": ["aria2c", "trim_galore", "cutadapt", "bismark", "deduplicate_bismark", "samtools", "bowtie2"]}


def _cfg():
    paths = load_paths()
    cfg = load_component("datasets", "gse149438")
    for k in ("meta_dir", "raw_dir", "processed_dir"):
        cfg[k] = resolve_path(cfg[k], paths)
    cfg["reference_dir"] = paths["reference_root"] / "hg38"
    cfg["cpg_cache"] = cfg["reference_dir"] / "cpg_keys_autosomes.npy"
    return cfg


def _need(step: str) -> None:
    missing = [t for t in REQUIRED_TOOLS[step] if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"step '{step}' needs {missing} on PATH (see README: bioinformatics environment)")


def _run(cmd: str, log) -> None:
    log.write(f"$ {cmd}\n"); log.flush()
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=subprocess.STDOUT, executable="/bin/bash")
    log.write(f"# {time.time() - t0:.0f} s\n"); log.flush()


def _index_ready(ref: Path) -> bool:
    """Bisulfite_Genome/ appears as soon as indexing starts: require the last file each bowtie2-build writes."""
    return all(any((ref / "Bisulfite_Genome" / f"{c}_conversion").glob(f"BS_{c}.rev.2.bt2*")) for c in ("CT", "GA"))


def _download(url: str, dest: Path, validate=None, tries: int = 3) -> None:
    """Download to <dest>.part, validate, then atomically move into place (an existing good file is never clobbered)."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        subprocess.run(["wget", "-q", "-O", str(part), url], check=True)
        if validate is None or validate(part):
            part.replace(dest); return
        print(f"invalid download of {url} (attempt {attempt})", file=sys.stderr)
    raise SystemExit(f"could not download a valid {dest.name} from {url}")


def _valid_ena(path: Path) -> bool:
    lines = path.read_text().splitlines()
    return len(lines) > 1 and "ERROR" not in path.read_text() and "fastq_md5" in lines[0]


def cmd_metadata(cfg, args) -> None:
    meta = cfg["meta_dir"]; meta.mkdir(parents=True, exist_ok=True)
    soft, ena = meta / "GSE149438_family.soft.gz", meta / "ena_runs.tsv"
    if not soft.exists():
        _download(ds.GEO_SOFT_URL, soft)
    if not ena.exists() or not _valid_ena(ena):
        _download(ds.ENA_FILEREPORT_URL, ena, _valid_ena)
    table = ds.build_sample_table(soft, ena)
    table.to_csv(meta / "samples_meta.tsv", sep="\t", index=False)
    print(table.groupby(["diagnosis", "batch"]).size().unstack(fill_value=0))
    print(f"{len(table)} samples, {table.fastq_bytes.sum() / 1e9:.1f} GB FASTQ -> {meta / 'samples_meta.tsv'}")


def cmd_reference(cfg, args) -> None:
    _need("reference")
    ref = cfg["reference_dir"]; ref.mkdir(parents=True, exist_ok=True)
    fa = ref / "hg38.fa.gz"
    if not fa.exists():
        subprocess.run(["wget", "-q", "-O", str(fa) + ".part", HG38_URL], check=True)
        Path(str(fa) + ".part").replace(fa)
    if not _index_ready(ref):
        subprocess.run(["bismark_genome_preparation", "--parallel", str(max(1, args.threads // 2)), str(ref)], check=True)
    print("reference ready:", ref)


def _process_one(row: dict, cfg: dict, threads: int, dry_run: bool, keep_bam: bool = False) -> dict:
    row = pd.Series(row)
    sid = row.sample_id
    out_frag = cfg["processed_dir"] / "fragments" / sid
    if (out_frag / "extract_stats.json").exists():
        return {"sample_id": sid, "status": "done"}
    pp = cfg["preprocessing"]
    raw = cfg["raw_dir"]
    cmds = ds.sample_commands(row, raw_dir=raw, bismark_index=cfg["reference_dir"], pp=pp, threads=threads)
    if dry_run:
        return {"sample_id": sid, "status": "dry-run", "commands": cmds}
    (raw / sid).mkdir(parents=True, exist_ok=True)
    bam = raw / sid / f"{sid}.dedup.bam"
    with open(raw / sid / "pipeline.log", "a") as log:
        if not bam.exists():
            for c in cmds:
                _run(c, log)
        t0 = time.time()
        off, keys, state, stats = extract_fragments(bam, contigs=cfg["contigs"], min_mapq=pp["min_mapq"],
                                                    require_proper_pair=pp["require_proper_pair"])
        # Bismark can call "CpG" context across an indel (a read CG spanning a deletion, typically in homopolymers);
        # such calls do not sit on a reference CpG and are dropped (counted), as wgbstools does
        ref = np.load(cfg["cpg_cache"], mmap_mode="r")
        off, keys, state, fstats = filter_calls(off, keys, state, np.isin(keys, ref))
        stats.update({f"not_reference_cpg_{k}": v for k, v in fstats.items()})
        counts = write_sample(out_frag, off, keys, state)
        stats.update(counts, extract_seconds=round(time.time() - t0, 1), finished_utc=datetime.now(timezone.utc).isoformat())
        (out_frag / "extract_stats.json").write_text(json.dumps(stats, indent=2))
    for p in ds.intermediate_files(raw, sid) + ([] if pp.get("keep_dedup_bam") or keep_bam else [bam]):
        p.unlink(missing_ok=True)
    return {"sample_id": sid, "status": "ok", **counts}


def cmd_process(cfg, args) -> None:
    meta = pd.read_csv(cfg["meta_dir"] / "samples_meta.tsv", sep="\t")
    if args.samples:
        unknown = set(args.samples) - set(meta.sample_id)
        if unknown:
            raise SystemExit(f"unknown samples: {sorted(unknown)}")
        meta = meta[meta.sample_id.isin(args.samples)]
    if args.limit:
        meta = meta.sort_values("fastq_bytes").head(args.limit)
    if not args.dry_run:
        _need("process")
        if not _index_ready(cfg["reference_dir"]):
            raise SystemExit("Bismark index missing: run the 'reference' step first")
        reference_cpg_keys(cfg["reference_dir"] / "hg38.fa.gz", cfg["cpg_cache"], cfg["contigs"])
    rows = meta.to_dict("records")
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_one, r, cfg, args.threads, args.dry_run, args.keep_bam): r["sample_id"] for r in rows}
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception:  # keep going; failures are listed at the end and the sample stays not-done
                failures.append(futs[f]); print(f"FAILED {futs[f]}\n{traceback.format_exc()}", file=sys.stderr)
                continue
            if args.dry_run:
                print(f"# {res['sample_id']}"); print("\n".join(res["commands"]))
            else:
                print(json.dumps(res), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} samples failed: {failures}")


def cmd_finalize(cfg, args) -> None:
    meta = pd.read_csv(cfg["meta_dir"] / "samples_meta.tsv", sep="\t")
    frag = cfg["processed_dir"] / "fragments"
    done = [s for s in meta.sample_id if (frag / s / "extract_stats.json").exists()]
    missing = sorted(set(meta.sample_id) - set(done))
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} samples not processed (e.g. {missing[:5]}); use --allow-partial for a "
                         "development build")
    stats = {s: json.loads((frag / s / "extract_stats.json").read_text()) for s in done}
    table = meta[meta.sample_id.isin(done)].copy()
    table["n_fragments"] = table.sample_id.map(lambda s: stats[s]["n_fragments"])
    table["n_cpgs"] = table.sample_id.map(lambda s: stats[s]["n_cpgs"])
    table = validate_samples(table)
    build = {"dataset": "gse149438", "genome_build": cfg["genome_build"], "contigs": cfg["contigs"],
             "preprocessing": cfg["preprocessing"], "partial": bool(missing), "missing_samples": missing,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "filter_totals": {k: int(sum(s.get(k, 0) for s in stats.values()))
                               for k in sorted({k for s in stats.values() for k in s if k.startswith(("reads_", "cpg_", "pairs_", "fragments_"))})}}
    manifest = finalize_dataset(cfg["processed_dir"], table.sample_id.tolist(), build)
    table.to_parquet(cfg["processed_dir"] / "samples.parquet", index=False)
    print(json.dumps({k: manifest[k] for k in ("n_samples", "n_loci", "partial")}, indent=2))


def cmd_status(cfg, args) -> None:
    meta_file = cfg["meta_dir"] / "samples_meta.tsv"
    if not meta_file.exists():
        print("metadata: MISSING"); return
    meta = pd.read_csv(meta_file, sep="\t")
    frag = cfg["processed_dir"] / "fragments"
    done = sum((frag / s / "extract_stats.json").exists() for s in meta.sample_id)
    print(f"metadata: {len(meta)} samples; processed: {done}/{len(meta)}; "
          f"reference: {'ok' if _index_ready(cfg['reference_dir']) else 'MISSING'}; "
          f"finalized: {(cfg['processed_dir'] / 'dataset_manifest.json').exists()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("metadata"); sub.add_parser("status")
    r = sub.add_parser("reference"); r.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("process")
    p.add_argument("--samples", nargs="*"); p.add_argument("--limit", type=int, default=0,
                                                           help="process the N smallest samples (pilot)")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--threads", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-bam", action="store_true", help="keep the dedup BAM (QC of a pilot sample)")
    f = sub.add_parser("finalize"); f.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    cfg = _cfg()
    {"metadata": cmd_metadata, "reference": cmd_reference, "process": cmd_process, "finalize": cmd_finalize,
     "status": cmd_status}[args.step](cfg, args)


if __name__ == "__main__":
    main()
