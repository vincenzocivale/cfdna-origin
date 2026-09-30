#!/usr/bin/env python3
"""GSE149438 CpG-per-fragment / storage audit of the processed fragment stores (docs/FRAGMENT_STORAGE_AUDIT.md).

For every processed sample (or --samples) reads fragments/<sid>/{frag_offsets,state}.npy and reports:
  * P(n CpG per fragment = k), k = 1..KMAX and a KMAX+ bin
  * for min_CpGs thresholds t (default 1..5): fragments and CpG calls retained (count, fraction), mean CpGs/fragment,
    % methylation of retained calls, fraction of fully methylated / fully unmethylated fragments
  * storage in the current format (4 B locus_row + 1 B state per call, 8 B int64 offset per fragment) and its
    extrapolation to all samples of samples_meta.tsv: bytes per raw read pair of each pilot sample x read_pairs of
    every sample (median ratio = point estimate; min/max ratio = range; also per-batch median ratio)
  * batch / class differences of per-sample retention (Mann-Whitney KRp1 vs KRp2 and Healthy vs cancer,
    Kruskal-Wallis across diagnoses). The pilot is tiny: p-values are descriptive only.

Nothing here chooses min_CpGs. Outputs -> {data_root}/gse149438/audits/fragment_storage_*.csv / .json

Example:
  python scripts/audit_fragment_storage.py
  python scripts/audit_fragment_storage.py --samples GSM4502214 GSM4502069 --thresholds 1 2 3 4 5 --kmax 40
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

BYTES_PER_CALL = 5     # int32 locus_row + uint8 state
BYTES_PER_FRAG = 8     # int64 frag_offsets entry


# ----------------------------------------------------------------------------------------------- pure helpers
def cpg_hist(n: np.ndarray, kmax: int = 40) -> np.ndarray:
    """Counts of fragments with k = 1..kmax CpGs; the last element is the kmax+ bin (so the length is kmax)."""
    n = np.asarray(n)
    if (n < 1).any():
        raise ValueError("fragments must have >= 1 CpG")
    h = np.bincount(np.minimum(n, kmax), minlength=kmax + 1)[1:]
    return h


def threshold_table(off: np.ndarray, state: np.ndarray, thresholds=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Per min_CpGs threshold: retained fragments/calls and methylation composition (CSR arrays of one sample)."""
    off = np.asarray(off, np.int64); state = np.asarray(state, np.uint8)
    if off[0] != 0 or off[-1] != len(state):
        raise ValueError("inconsistent CSR arrays")
    n = np.diff(off)
    if (n < 1).any():
        raise ValueError("empty fragments")
    meth = np.add.reduceat(state.astype(np.int64), off[:-1]) if len(n) else np.zeros(0, np.int64)
    F, C = len(n), int(n.sum())
    rows = []
    for t in thresholds:
        k = n >= t
        f, c, m = int(k.sum()), int(n[k].sum()), int(meth[k].sum())
        rows.append({"min_cpgs": t, "fragments": f, "fragments_frac": f / F if F else np.nan, "calls": c,
                     "calls_frac": c / C if C else np.nan, "mean_cpgs_per_fragment": c / f if f else np.nan,
                     "pct_meth": 100 * m / c if c else np.nan,
                     "frac_fully_meth": float((meth[k] == n[k]).mean()) if f else np.nan,
                     "frac_fully_unmeth": float((meth[k] == 0).mean()) if f else np.nan,
                     "bytes": storage_bytes(c, f)})
    return pd.DataFrame(rows)


def parse_dedup_report(text: str) -> dict:
    """deduplicate_bismark report -> {'aligned_pairs', 'duplicates_removed', 'dup_rate'} (empty if unparsable)."""
    import re

    a = re.search(r"Total number of alignments analysed in .*?:\s*(\d+)", text)
    d = re.search(r"Total number duplicated alignments removed:\s*(\d+)", text)
    if not (a and d):
        return {}
    a, d = int(a.group(1)), int(d.group(1))
    return {"aligned_pairs": a, "duplicates_removed": d, "dup_rate": d / a if a else float("nan")}


def storage_bytes(calls: int, fragments: int) -> int:
    """Current on-disk format of one sample: 5 B per call + 8 B per offset (F + 1 offsets)."""
    return BYTES_PER_CALL * int(calls) + BYTES_PER_FRAG * (int(fragments) + 1)


def extrapolate(per: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """per: one row per (sample_id, min_cpgs) with 'bytes' and 'read_pairs'; meta: all samples with read_pairs,
    batch. Total bytes over meta = sum(read_pairs) x (bytes / read_pair ratio of the pilot): median (point), min, max;
    and per batch with that batch's pilot median ratio."""
    per = per.assign(ratio=per.bytes / per.read_pairs)
    out = []
    for t, g in per.groupby("min_cpgs"):
        rp = meta.read_pairs.sum()
        byb = sum(meta[meta.batch == b].read_pairs.sum() * (g[g.batch == b].ratio.median()
                                                            if (g.batch == b).any() else g.ratio.median())
                  for b in meta.batch.unique())
        out.append({"min_cpgs": t, "n_pilot": len(g), "n_target": len(meta), "bytes_per_read_pair_median":
                    g.ratio.median(), "total_GB_median": rp * g.ratio.median() / 1e9,
                    "total_GB_min": rp * g.ratio.min() / 1e9, "total_GB_max": rp * g.ratio.max() / 1e9,
                    "total_GB_batch_median": byb / 1e9})
    return pd.DataFrame(out)


def group_tests(per: pd.DataFrame, value: str) -> dict:
    """Descriptive tests of one per-sample value: KRp1 vs KRp2, Healthy vs cancer (all, and within each batch),
    Kruskal-Wallis across diagnoses."""
    from scipy import stats as ss

    def mw(a, b):
        return float(ss.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else float("nan")

    healthy = per.diagnosis == "Normal"
    res = {"batch_KRp1_mean": float(per[per.batch == "KRp1"][value].mean()),
           "batch_KRp2_mean": float(per[per.batch == "KRp2"][value].mean()),
           "batch_mannwhitney_p": mw(per[per.batch == "KRp1"][value], per[per.batch == "KRp2"][value]),
           "healthy_mean": float(per[healthy][value].mean()), "cancer_mean": float(per[~healthy][value].mean()),
           "healthy_vs_cancer_mannwhitney_p": mw(per[healthy][value], per[~healthy][value])}
    for b in sorted(per.batch.unique()):
        s = per[per.batch == b]
        res[f"healthy_vs_cancer_within_{b}_p"] = mw(s[s.diagnosis == "Normal"][value], s[s.diagnosis != "Normal"][value])
    groups = [g[value].values for _, g in per.groupby("diagnosis")]
    res["kruskal_diagnosis_p"] = float(ss.kruskal(*groups).pvalue) if len(groups) > 1 else float("nan")
    res["per_diagnosis_mean"] = per.groupby("diagnosis")[value].mean().to_dict()
    return res


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="*", help="default: every sample with extract_stats.json")
    ap.add_argument("--thresholds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--kmax", type=int, default=40)
    args = ap.parse_args()

    from cfdna_origin.config import load_component, load_paths, resolve_path

    paths = load_paths()
    cfg = load_component("datasets", "gse149438")
    meta = pd.read_csv(resolve_path(cfg["meta_dir"], paths) / "samples_meta.tsv", sep="\t")
    frag = resolve_path(cfg["processed_dir"], paths) / "fragments"
    raw = resolve_path(cfg["raw_dir"], paths)
    out = paths["data_root"] / "gse149438" / "audits"; out.mkdir(parents=True, exist_ok=True)
    sids = args.samples or sorted(s for s in meta.sample_id if (frag / s / "extract_stats.json").exists())
    if not sids:
        raise SystemExit("no processed samples")
    m = meta.set_index("sample_id")
    per_rows, hist_rows = [], []
    for sid in sids:
        d = frag / sid
        off = np.load(d / "frag_offsets.npy"); state = np.load(d / "state.npy", mmap_mode="r")
        info = {"sample_id": sid, "diagnosis": m.loc[sid, "diagnosis"], "batch": m.loc[sid, "batch"],
                "read_pairs": int(m.loc[sid, "read_pairs"])}
        rep = sorted((raw / sid).glob("*deduplication_report.txt"))
        info.update(parse_dedup_report(rep[0].read_text()) if rep else {})
        tt = threshold_table(off, state, args.thresholds).assign(**info)
        per_rows.append(tt)
        h = cpg_hist(np.diff(off), args.kmax)
        hist_rows.append(pd.DataFrame({"k": np.arange(1, args.kmax + 1), "fragments": h, "p": h / h.sum(),
                                       "k_label": [str(k) for k in range(1, args.kmax)] + [f"{args.kmax}+"]}
                                      ).assign(**info))
    per = pd.concat(per_rows, ignore_index=True); hist = pd.concat(hist_rows, ignore_index=True)
    per["fragments_per_read_pair"] = per.fragments / per.read_pairs
    per.to_csv(out / "fragment_storage_per_sample.csv", index=False)
    hist.to_csv(out / "fragment_storage_cpg_hist.csv", index=False)
    pooled_hist = hist.groupby(["k", "k_label"]).fragments.sum().reset_index()
    pooled_hist["p"] = pooled_hist.fragments / pooled_hist.fragments.sum()
    ext = extrapolate(per, meta)
    agg = per.groupby("min_cpgs").agg(fragments_frac_mean=("fragments_frac", "mean"),
                                      fragments_frac_min=("fragments_frac", "min"),
                                      fragments_frac_max=("fragments_frac", "max"),
                                      calls_frac_mean=("calls_frac", "mean"), fragments_total=("fragments", "sum"),
                                      calls_total=("calls", "sum"), mean_cpgs=("mean_cpgs_per_fragment", "mean"),
                                      pct_meth=("pct_meth", "mean"), fully_meth=("frac_fully_meth", "mean"),
                                      fully_unmeth=("frac_fully_unmeth", "mean"),
                                      pilot_bytes=("bytes", "sum")).reset_index().merge(ext, on="min_cpgs")
    agg.to_csv(out / "fragment_storage_thresholds.csv", index=False)
    by_group = per.groupby(["min_cpgs", "batch", "diagnosis"])[
        ["fragments_frac", "calls_frac", "mean_cpgs_per_fragment", "pct_meth", "frac_fully_meth",
         "frac_fully_unmeth"]].mean().reset_index()
    by_group.to_csv(out / "fragment_storage_by_group.csv", index=False)
    tests = {f"min{t}": {v: group_tests(per[per.min_cpgs == t], v)
                         for v in ("fragments_frac", "calls_frac", "pct_meth", "frac_fully_meth",
                                   "frac_fully_unmeth", "mean_cpgs_per_fragment", "fragments_per_read_pair")
                         + (("dup_rate",) if "dup_rate" in per else ())}
             for t in args.thresholds}
    summary = {"samples": sids, "n": len(sids), "format": {"bytes_per_call": BYTES_PER_CALL,
                                                           "bytes_per_fragment": BYTES_PER_FRAG},
               "extrapolation": "total = sum(read_pairs over samples_meta.tsv) x pilot bytes/read_pair (median; "
                                "min/max = range; batch_median uses each batch's pilot median)",
               "pooled_hist": pooled_hist.round(6).to_dict("records"),
               "thresholds": agg.round(6).to_dict("records"), "tests": tests}
    (out / "fragment_storage_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
