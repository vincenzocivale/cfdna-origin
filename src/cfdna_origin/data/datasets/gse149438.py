"""GSE149438 / PRJNA628686 — EpiPanGI Dx (Kandimalla et al., Clin Cancer Res 2021;27:6135).

300 plasma cfDNA samples, one patient each (GEO `alternate id`, e.g. HCC_43). Targeted bisulfite capture
(NimbleGen SeqCap Epi, 67,832 DMRs / 25.6 Mb; Swift Accel-Methyl libraries; NovaSeq paired-end). GEO provides only
per-CpG methratio tables (hg19), so read-level patterns require re-aligning the public FASTQ (ENA) to GRCh38.

Diagnoses as published: Normal, CRC, PDAC, HCC, GC, ESCC, EAC. The paper merged ESCC+EAC for its multi-cancer model;
both label schemes are provided (`configs/datasets/gse149438.yaml`).

Known confounder: the GEO title prefix KRp1/KRp2 (probable library/sequencing batch) is strongly aliased with the
class (CRC 40/40 KRp2; Normal 37/46 KRp1). It is kept as `batch` for stratification and per-batch reporting.
"""
from __future__ import annotations

import gzip
import shlex
from pathlib import Path

import pandas as pd

GEO_SOFT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149438/soft/GSE149438_family.soft.gz"
ENA_FILEREPORT_URL = (
    "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA628686&result=read_run&fields="
    "run_accession,sample_alias,sample_title,library_layout,read_count,base_count,fastq_ftp,fastq_md5,fastq_bytes"
    "&format=tsv"
)


def parse_soft(path: Path) -> pd.DataFrame:
    recs, cur = [], None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                cur = {"gsm": line.split("=")[1].strip()}; recs.append(cur)
            elif cur is not None and line.startswith("!Sample_"):
                key, _, value = line.partition(" = ")
                key = key[len("!Sample_"):]
                if key == "characteristics_ch1":
                    k, _, v = value.partition(": ")
                    cur["char_" + k.strip().lower().replace(" ", "_")] = v.strip()
                elif key in ("title", "source_name_ch1"):
                    cur[key] = value
    return pd.DataFrame(recs)


def build_sample_table(soft_path: Path, ena_path: Path) -> pd.DataFrame:
    """Clean metadata table (one row per sample) from the GEO SOFT and the ENA run report. No data-derived fields."""
    soft = parse_soft(soft_path)
    ena = pd.read_csv(ena_path, sep="\t")
    if "fastq_md5" not in ena:
        ena["fastq_md5"] = ""
    runs = ena.groupby("sample_alias").agg(
        runs=("run_accession", lambda x: ";".join(sorted(x))),
        fastq_ftp=("fastq_ftp", lambda x: ";".join(x)),
        fastq_md5=("fastq_md5", lambda x: ";".join(x.fillna("").astype(str))),
        read_pairs=("read_count", "sum"),
        fastq_bytes=("fastq_bytes", lambda x: sum(int(b) for v in x for b in str(v).split(";"))),
    ).reset_index().rename(columns={"sample_alias": "gsm"})
    df = soft.merge(runs, on="gsm", how="left", validate="one_to_one")
    if df.runs.isna().any():
        raise ValueError(f"samples without ENA runs: {df.gsm[df.runs.isna()].tolist()}")
    out = pd.DataFrame({
        "sample_id": df.gsm,
        "patient_id": df.char_alternate_id,
        "diagnosis": df.char_tissue,
        "sample_type": "plasma_cfdna",
        "batch": df.title.str.extract(r"^([A-Za-z]+\d*)-")[0],
        "age": pd.to_numeric(df.get("char_age"), errors="coerce"),
        "stage": df.get("char_stage"),
        "title": df.title,
        "source_accession": df.runs,
        "fastq_ftp": df.fastq_ftp,
        "fastq_md5": df.fastq_md5,
        "read_pairs": df.read_pairs,
        "fastq_bytes": df.fastq_bytes,
    })
    if out.patient_id.duplicated().any():
        raise ValueError("GSE149438 is expected to have one sample per patient")
    return out


def sample_commands(row: pd.Series, *, raw_dir: Path, bismark_index: Path, pp: dict, threads: int) -> list[str]:
    """Shell commands (download -> trim -> Bismark -> dedup) for one sample. Output: <raw_dir>/<sid>/<sid>.dedup.bam."""
    sid = row.sample_id
    d = raw_dir / sid
    urls = [u for u in str(row.fastq_ftp).split(";") if u]
    if len(urls) != 2:
        raise ValueError(f"{sid}: expected 2 paired FASTQ files, got {urls}")
    md5 = str(row.fastq_md5).split(";")
    q = shlex.quote
    cmds = [f"mkdir -p {q(str(d))}"]
    for i, (u, m) in enumerate(zip(urls, md5 + [""] * 2), start=1):
        fq = d / f"{sid}_R{i}.fastq.gz"
        cmds.append(f"aria2c -q -x 8 -s 8 --allow-overwrite=true -d {q(str(d))} -o {q(fq.name)} https://{u}")
        if m and m != "nan":
            cmds.append(f"echo '{m}  {fq}' | md5sum -c --quiet -")
    t = pp["trim"]
    cmds.append(
        f"trim_galore --paired --cores {min(threads, 8)} --clip_R1 {t['clip_r1']} --clip_R2 {t['clip_r2']} "
        f"--three_prime_clip_R1 {t['three_prime_clip_r1']} --three_prime_clip_R2 {t['three_prime_clip_r2']} "
        f"-o {q(str(d))} {q(str(d / f'{sid}_R1.fastq.gz'))} {q(str(d / f'{sid}_R2.fastq.gz'))}"
    )
    b = pp["bismark"]
    aln = _bismark_stem(sid)  # Bismark rejects --basename with --parallel: outputs are named after the R1 file
    cmds.append(
        f"bismark --genome {q(str(bismark_index))} --parallel {b['parallel']} {b.get('extra', '')} "
        f"--temp_dir {q(str(d))} -o {q(str(d))} "
        f"-1 {q(str(d / f'{sid}_R1_val_1.fq.gz'))} -2 {q(str(d / f'{sid}_R2_val_2.fq.gz'))}"
    )
    cmds.append(f"deduplicate_bismark -p --bam --output_dir {q(str(d))} {q(str(d / f'{aln}.bam'))}")
    cmds.append(f"mv {q(str(d / f'{aln}.deduplicated.bam'))} {q(str(d / f'{sid}.dedup.bam'))}")
    return cmds


def _bismark_stem(sid: str) -> str:
    return f"{sid}_R1_val_1_bismark_bt2_pe"


def intermediate_files(raw_dir: Path, sid: str) -> list[Path]:
    """Large intermediates deleted after a sample's fragments are written (the dedup BAM is kept unless asked)."""
    d = raw_dir / sid
    return [d / f"{sid}_R1.fastq.gz", d / f"{sid}_R2.fastq.gz", d / f"{sid}_R1_val_1.fq.gz",
            d / f"{sid}_R2_val_2.fq.gz", d / f"{_bismark_stem(sid)}.bam"]
