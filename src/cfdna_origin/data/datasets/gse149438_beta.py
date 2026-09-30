"""GSE149438 processed per-CpG methylation (`gse149438_processed_beta` data mode).

Source: GEO series supplementary `GSE149438_RAW.tar` = 300 `GSM*_KRp{1,2}-N_bsmap_meth.txt.gz`, BSMAP methratio.py
output on **hg19**, one row per CpG with both strands collapsed onto the `+` strand C (1-based position):
    chr pos strand context ratio eff_CT_count C_count CT_count rev_G_count rev_GA_count CI_lower CI_upper
Only CG context, autosomes, CT_count >= 4 are present in the published files (verified on the files themselves).
beta = C_count / CT_count; coverage = CT_count.

Processing (no imputation, no synthetic values):
  1. per sample: keep rows with context CG, autosomes, coverage >= `min_coverage`;
  2. union of observed hg19 loci -> chain liftover to GRCh38 (unmapped and ambiguous mappings dropped), target must be
     a CpG of the GRCh38 reference, many-to-one collisions dropped, target must be in every configured representation
     store's universe; every drop is counted in the manifest;
  3. dense matrices over the final loci: beta float16 (NaN = not observed in that sample), coverage uint16 (0 = not
     observed). Missing CpGs stay missing: the models only see observed (locus, beta, coverage) tokens.
"""
from __future__ import annotations

import gzip
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from cfdna_origin.data.loci import AUTOSOMES, CHROM_CODE

MEMBER_RE = re.compile(r"^(GSM\d+)_(KRp\d)-(\d+)_bsmap_meth\.txt\.gz$")
COLUMNS = ["chr", "pos", "strand", "context", "ratio", "eff_CT_count", "C_count", "CT_count", "rev_G_count",
           "rev_GA_count", "CI_lower", "CI_upper"]


def list_members(tar_path: Path) -> dict[str, str]:
    """GSM -> member name."""
    with tarfile.open(tar_path) as tar:
        out = {}
        for m in tar.getmembers():
            hit = MEMBER_RE.match(Path(m.name).name)
            if hit:
                out[hit.group(1)] = m.name
    return out


def parse_methratio(handle, min_coverage: int, contigs: list[str] = AUTOSOMES) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(hg19 locus keys, C counts, CT counts, stats) from one methratio table (text or gzip handle)."""
    df = pd.read_csv(handle, sep="\t", usecols=["chr", "pos", "strand", "context", "C_count", "CT_count"],
                     dtype={"chr": str, "pos": np.int64, "strand": str, "context": str, "C_count": np.int64,
                            "CT_count": np.int64})
    stats = {"rows": int(len(df))}
    keep = (df.context == "CG").to_numpy()
    stats["dropped_non_cg"] = int((~keep).sum())
    on = df.chr.isin(contigs).to_numpy()
    stats["dropped_other_contig"] = int((keep & ~on).sum())
    keep &= on
    cov_ok = (df.CT_count >= min_coverage).to_numpy()
    stats["dropped_low_coverage"] = int((keep & ~cov_ok).sum())
    keep &= cov_ok
    if (df.strand[keep] != "+").any():
        raise ValueError("expected strand-collapsed '+' rows only")
    df = df[keep]
    if (df.C_count > df.CT_count).any():
        raise ValueError("C_count > CT_count")
    codes = df.chr.map(CHROM_CODE).to_numpy(np.int64)
    keys = (codes << np.int64(32)) | df.pos.to_numpy(np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError("duplicated CpG rows")
    stats["kept"] = int(len(keys))
    return keys, df.C_count.to_numpy(np.int64)[order], df.CT_count.to_numpy(np.int64)[order], stats


def read_member(tar_path: Path, member: str, min_coverage: int):
    with tarfile.open(tar_path) as tar:
        raw = tar.extractfile(member).read()
    with gzip.open(io.BytesIO(raw), "rt") as handle:
        return parse_methratio(handle, min_coverage)


def resolve_collisions(target: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop every source locus whose (valid) target is shared with another source locus (many-to-one)."""
    t = np.where(ok, target, -1)
    valid_t = t[ok]
    uniq, counts = np.unique(valid_t, return_counts=True)
    dup = uniq[counts > 1]
    clash = ok & np.isin(t, dup)
    return ok & ~clash, int(clash.sum())


def sample_summary(beta_row: np.ndarray, cov_row: np.ndarray) -> dict:
    obs = cov_row > 0
    b = beta_row[obs].astype(np.float64)
    c = cov_row[obs].astype(np.float64)
    return {"n_observed_loci": int(obs.sum()), "missing_fraction": float(1 - obs.mean()),
            "mean_beta": float(b.mean()) if len(b) else np.nan, "beta_variance": float(b.var()) if len(b) else np.nan,
            "mean_coverage": float(c.mean()) if len(c) else np.nan,
            "median_coverage": float(np.median(c)) if len(c) else np.nan}


def build_samples_table(meta: pd.DataFrame, members: dict[str, str]) -> pd.DataFrame:
    missing = sorted(set(meta.sample_id) - set(members))
    if missing:
        raise FileNotFoundError(f"{len(missing)} samples have no methratio file in the tar: {missing[:5]}")
    extra = sorted(set(members) - set(meta.sample_id))
    if extra:
        raise ValueError(f"methratio files without metadata: {extra[:5]}")
    df = meta.copy()
    df["source_file"] = df.sample_id.map(members)
    title_batch = df.source_file.map(lambda m: MEMBER_RE.match(Path(m).name).group(2))
    if (title_batch != df.batch).any():
        raise ValueError("batch in the file name disagrees with the GEO title prefix")
    return df
