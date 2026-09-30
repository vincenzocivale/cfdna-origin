#!/usr/bin/env python3
"""GSE149438 M-bias / trimming audit on a small raw-read pilot (docs/MBIAS_AUDIT.md).

Steps (per sample; each writes under {data_root}/gse149438/audits/mbias/):
  bismark   `bismark_methylation_extractor -p --mbias_only --include_overlap` on the dedup BAM (a name-sorted copy is
            made first if the BAM is coordinate-sorted) -> bismark/<sid>.M-bias.txt           [needs bismark, samtools]
  scan      one pysam pass over the dedup BAM with the fragment extractor's read filters -> scan/<sid>.npz:
              * CpG M-bias per read (R1/R2) by position from the read 5' end AND from the read 3' end
              * CpG methylation by distance (bp) from each fragment end (R1-5' end, R2-5' end), per source read
              * fragment-level simulation of extra clipping (unique merged CpG calls, methylated calls, mate
                conflicts, fragments with >= k CpGs) for every (clip_r1, clip_r2, extra 3' clip) in the grid
  validate  re-run trim -> Bismark -> dedup for ONE sample with a different clip_r1 (e.g. 20) under audits/work/, then
            scan it, and compare observed vs simulated counts                     [needs aria2c, trim_galore, bismark]
  report    aggregate -> per-sample / per-batch / per-class CSVs + summary.json

Positions: the dedup BAM already lacks the first clip_r1 (R1) / clip_r2 (R2) sequenced cycles, so read position p
(1-based) = sequencing cycle p + clip. Extra 5' clipping of e bases is simulated by discarding the calls at read
positions <= e (alignment changes are ignored; `validate` measures that approximation).

Example:
  python scripts/audit_mbias.py bismark --samples GSM4502214 --threads 4
  python scripts/audit_mbias.py scan --samples GSM4502214 --workers 10
  python scripts/audit_mbias.py validate --sample GSM4502214 --clip-r1 20 --threads 16
  python scripts/audit_mbias.py report
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

MAXPOS = 200           # read positions tracked (reads are <= 151 bp)
MAXDIST = 600          # fragment-end distances tracked (Bismark --maxins 500)
CLIP_R1 = (10, 15, 20, 25)   # total 5' clip of R1 simulated (the BAM has 10)
CLIP_R2 = (15, 20)           # total 5' clip of R2 simulated (the BAM has 15)
EXTRA3 = (0, 5)              # extra 3' clip of both reads on top of three_prime_clip 10/10
KMAX = 5                     # fragments with >= 1..KMAX CpGs are counted per scenario
_BAD_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
_CALL = re.compile("[Zz]")
_QREF, _QONLY, _RONLY = (0, 7, 8), (1, 4), (2, 3)


# ----------------------------------------------------------------------------------------------- pure helpers
def parse_mbias_txt(path_or_text) -> pd.DataFrame:
    """Bismark *M-bias.txt -> long table (context, read, position, meth, unmeth, pct, coverage)."""
    text = Path(path_or_text).read_text() if not isinstance(path_or_text, str) or "\n" not in path_or_text \
        else path_or_text
    rows, ctx, read = [], None, None
    for line in text.splitlines():
        m = re.match(r"^(CpG|CHG|CHH) context \((R[12])\)", line)
        if m:
            ctx, read = m.group(1), int(m.group(2)[1]); continue
        f = line.split("\t")
        if ctx and len(f) == 5 and f[0].isdigit():
            rows.append((ctx, read, int(f[0]), int(f[1]), int(f[2]), float(f[3]) if f[3] else np.nan, int(f[4])))
    if not rows:
        raise ValueError("no M-bias rows parsed")
    return pd.DataFrame(rows, columns=["context", "read", "position", "meth", "unmeth", "pct", "coverage"])


def read_cpg_calls(xm: str, xg: str, reference_start: int, cigartuples, is_reverse: bool):
    """CpG calls of one aligned read as (1-based forward C position, state, position from read 5' end (0-based),
    read length). Same coordinate logic as cfdna_origin.data.bam_fragments.read_calls; calls inside insertions or
    soft clips are dropped. The read 5' end is the XM end for forward reads and the XM start for reverse reads."""
    L = len(xm)
    hits = [m.start() for m in _CALL.finditer(xm)]
    if not hits:
        return [], L
    base = reference_start + (1 if xg == "CT" else 0)
    out = []
    if len(cigartuples) == 1 and cigartuples[0][0] in _QREF:
        for h in hits:
            out.append((base + h, int(xm[h] == "Z"), L - 1 - h if is_reverse else h))
        return out, L
    qs, rs, ls = [], [], []
    q = r = 0
    for op, n in cigartuples:
        if op in _QREF:
            qs.append(q); rs.append(r); ls.append(n); q += n; r += n
        elif op in _QONLY:
            q += n
        elif op in _RONLY:
            r += n
    for h in hits:
        i = bisect_right(qs, h) - 1
        if i >= 0 and h < qs[i] + ls[i]:
            out.append((base + rs[i] + h - qs[i], int(xm[h] == "Z"), L - 1 - h if is_reverse else h))
    return out, L


def scenarios(clip_r1=CLIP_R1, clip_r2=CLIP_R2, extra3=EXTRA3, base_r1=10, base_r2=15):
    """(clip_r1, clip_r2, extra3, extra5_r1, extra5_r2) for every simulated trimming option."""
    return [(c1, c2, e3, c1 - base_r1, c2 - base_r2) for c1 in clip_r1 for c2 in clip_r2 for e3 in extra3]


def merge_pair(r1, l1, r2, l2, extra5_r1, extra5_r2, extra3):
    """Fragment-level merge after simulated extra clipping, as the extractor does (union; discordant CpG dropped).
    r1/r2: lists of (pos, state, pos5); l1/l2: read lengths. Returns (n_calls, n_meth, n_conflict)."""
    calls, conflict = {}, set()
    for calls_r, L, e5 in ((r1, l1, extra5_r1), (r2, l2, extra5_r2)):
        for pos, st, p5 in calls_r:
            if p5 < e5 or (L - 1 - p5) < extra3:
                continue
            if calls.setdefault(pos, st) != st:
                conflict.add(pos)
    for p in conflict:
        del calls[p]
    return len(calls), sum(calls.values()), len(conflict)


class ScanAccumulator:
    """Counts accumulated over read pairs (see module docstring)."""

    def __init__(self, scen=None):
        self.scen = scen if scen is not None else scenarios()
        self.m5 = np.zeros((2, 2, MAXPOS), np.int64)   # [read-1, state, pos from 5']
        self.m3 = np.zeros((2, 2, MAXPOS), np.int64)   # [read-1, state, pos from 3']
        self.fe = np.zeros((2, 2, 2, MAXDIST), np.int64)  # [end: 0=R1-5' 1=R2-5', source read-1, state, dist]
        self.sc = np.zeros((len(self.scen), 3 + KMAX), np.int64)  # calls, meth, conflicts, frags>=1..KMAX
        self.pairs = 0

    def add_pair(self, r1, l1, fiveprime1, r2, l2, fiveprime2):
        """r*: (pos, state, pos5) calls; fiveprime*: 1-based genomic coordinate of each read's 5'-most base."""
        self.pairs += 1
        for ri, (calls, L) in enumerate(((r1, l1), (r2, l2))):
            for pos, st, p5 in calls:
                if p5 < MAXPOS:
                    self.m5[ri, st, p5] += 1
                p3 = L - 1 - p5
                if 0 <= p3 < MAXPOS:
                    self.m3[ri, st, p3] += 1
                for end, fp in enumerate((fiveprime1, fiveprime2)):
                    d = abs(pos - fp)
                    if d < MAXDIST:
                        self.fe[end, ri, st, d] += 1
        for i, (_, _, e3, e1, e2) in enumerate(self.scen):
            n, m, c = merge_pair(r1, l1, r2, l2, e1, e2, e3)
            row = self.sc[i]
            row[0] += n; row[1] += m; row[2] += c
            for k in range(1, min(n, KMAX) + 1):
                row[2 + k] += 1

    def save(self, path: Path, meta: dict) -> None:
        np.savez_compressed(path, m5=self.m5, m3=self.m3, fe=self.fe, sc=self.sc,
                            scen=np.asarray(self.scen, np.int64), meta=json.dumps({**meta, "pairs": self.pairs}))


def mbias_frame(m: np.ndarray, clip_r1: int = 10, clip_r2: int = 15, end: str = "5p") -> pd.DataFrame:
    """[read, state, pos] counts -> long table with sequencing cycle (5' profile only: cycle = pos + clip)."""
    rows = []
    for ri, clip in ((0, clip_r1), (1, clip_r2)):
        u, mt = m[ri, 0], m[ri, 1]
        for p in range(m.shape[2]):
            cov = int(u[p] + mt[p])
            if cov:
                rows.append({"read": ri + 1, "end": end, "position": p + 1,
                             "cycle": p + 1 + clip if end == "5p" else np.nan,
                             "meth": int(mt[p]), "unmeth": int(u[p]), "coverage": cov, "pct": 100 * mt[p] / cov})
    return pd.DataFrame(rows)


def clip_metrics(prof: pd.DataFrame, base_clip: int, options, plateau_cycles=(40, 100), window: int = 10,
                 total_calls: int | None = None, min_cov: int = 1) -> pd.DataFrame:
    """For one read's 5' M-bias profile (columns position, cycle, meth, unmeth, coverage, pct), evaluate total 5'
    clip options: calls lost (fraction of this read's calls, and of `total_calls` if given), global % methylation of
    the retained calls and its change, plateau = median pct over cycles in plateau_cycles, residual deviation =
    mean |pct - plateau| over the first `window` retained cycles, and signed deviation over the same window."""
    prof = prof[prof.coverage >= min_cov].sort_values("position")
    pl = prof[(prof.cycle >= plateau_cycles[0]) & (prof.cycle <= plateau_cycles[1])]
    plateau = float(np.median(pl.pct)) if len(pl) else np.nan
    tot = prof.coverage.sum()
    base_pct = 100 * prof.meth.sum() / tot
    out = []
    for c in options:
        e = c - base_clip
        kept = prof[prof.position > e]
        lost = prof[prof.position <= e].coverage.sum()
        first = kept.head(window)
        pct = 100 * kept.meth.sum() / kept.coverage.sum()
        out.append({"clip": c, "extra": e, "calls_lost_frac_read": lost / tot,
                    "calls_lost_frac_all": lost / total_calls if total_calls else np.nan,
                    "pct_meth_read": pct, "delta_pct_meth_read": pct - base_pct, "plateau": plateau,
                    "resid_abs_dev": float(np.mean(np.abs(first.pct - plateau))),
                    "resid_signed_dev": float(np.mean(first.pct - plateau)),
                    "first_cycle_dev": float(first.pct.iloc[0] - plateau)})
    return pd.DataFrame(out)


def scenario_frame(sc: np.ndarray, scen: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(scen, columns=["clip_r1", "clip_r2", "extra3", "extra5_r1", "extra5_r2"])
    df["calls"], df["meth"], df["conflicts"] = sc[:, 0], sc[:, 1], sc[:, 2]
    for k in range(1, sc.shape[1] - 2):
        df[f"frags_ge{k}"] = sc[:, 2 + k]
    base = df[(df.extra5_r1 == 0) & (df.extra5_r2 == 0) & (df.extra3 == 0)].iloc[0]
    df["pct_meth"] = 100 * df.meth / df.calls
    df["calls_lost_frac"] = 1 - df.calls / base.calls
    df["delta_pct_meth"] = df.pct_meth - 100 * base.meth / base.calls
    for k in range(1, sc.shape[1] - 2):
        df[f"frags_ge{k}_lost_frac"] = 1 - df[f"frags_ge{k}"] / base[f"frags_ge{k}"]
    return df


def fragment_end_excess(fe: pd.DataFrame, near=(0, 9), far=(40, 100)) -> pd.DataFrame:
    """Per sample, end (R1_5p / R2_5p) and source read: % methylation of calls within `near` bp of that fragment end
    minus % methylation within `far` bp. If the excess near the R1-5' end is also seen in R2's calls (read at late R2
    cycles), the bias follows the molecule end, not the sequencing cycle. Columns excess_<end>_R<source>."""
    rows = []
    for sid, g in fe.groupby("sample_id"):
        r = {"sample_id": sid, "batch": g.batch.iloc[0], "diagnosis": g.diagnosis.iloc[0]}
        for (end, src), h in g.groupby(["end", "source_read"]):
            n = h[(h.dist >= near[0]) & (h.dist <= near[1])]
            f = h[(h.dist >= far[0]) & (h.dist <= far[1])]
            r[f"excess_{end}_R{src}"] = 100 * (n.meth.sum() / max(n.coverage.sum(), 1)
                                               - f.meth.sum() / max(f.coverage.sum(), 1))
        rows.append(r)
    return pd.DataFrame(rows)


def noncpg_excess(b: pd.DataFrame, first: int = 5, plateau_pos=(30, 85)) -> dict:
    """Bismark M-bias table -> CHH/CHG % methylation at the first `first` positions minus the median over
    `plateau_pos` (positions, not cycles), per read: a conversion / fill-in artifact indicator."""
    out = {}
    for ctx in ("CHH", "CHG", "CpG"):
        for r in (1, 2):
            t = b[(b.context == ctx) & (b.read == r)].sort_values("position")
            pl = t[(t.position >= plateau_pos[0]) & (t.position <= plateau_pos[1])].pct.median()
            out[f"{ctx}_R{r}_first{first}_excess"] = float(t.head(first).pct.mean() - pl)
    return out


# ----------------------------------------------------------------------------------------------- config / IO
def _cfg():
    from cfdna_origin.config import load_component, load_paths, resolve_path

    paths = load_paths()
    cfg = load_component("datasets", "gse149438")
    for k in ("meta_dir", "raw_dir", "processed_dir"):
        cfg[k] = resolve_path(cfg[k], paths)
    cfg["reference_dir"] = paths["reference_root"] / "hg38"
    cfg["audit_dir"] = paths["data_root"] / "gse149438" / "audits"
    cfg["mbias_dir"] = cfg["audit_dir"] / "mbias"
    return cfg


def _meta(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg["meta_dir"] / "samples_meta.tsv", sep="\t")


def _bam(cfg, sid: str, bam: Path | None = None) -> Path:
    bam = bam or cfg["raw_dir"] / sid / f"{sid}.dedup.bam"
    if not bam.exists():
        raise SystemExit(f"{bam} missing (process the sample with --keep-bam first)")
    return bam


def _sort_order(bam: Path) -> str:
    out = subprocess.run(["samtools", "view", "-H", str(bam)], check=True, capture_output=True, text=True).stdout
    m = re.search(r"^@HD.*\tSO:(\S+)", out, re.M)
    return m.group(1) if m else "unknown"


def _run(cmd: str, log) -> float:
    log.write(f"$ {cmd}\n"); log.flush()
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=subprocess.STDOUT, executable="/bin/bash")
    log.write(f"# {time.time() - t0:.0f} s\n"); log.flush()
    return time.time() - t0


# ----------------------------------------------------------------------------------------------- steps
def cmd_bismark(cfg, args) -> None:
    for t in ("bismark_methylation_extractor", "samtools"):
        if shutil.which(t) is None:
            raise SystemExit(f"{t} not on PATH")
    outdir = cfg["mbias_dir"] / "bismark"; outdir.mkdir(parents=True, exist_ok=True)
    for sid in args.samples:
        dest = outdir / f"{sid}.M-bias.txt"
        if dest.exists() and not args.force:
            print(f"{sid}: done"); continue
        bam = _bam(cfg, sid)
        work = cfg["audit_dir"] / "work" / sid; work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(work / "mbias.log", "a") as log:
            src = bam
            if _sort_order(bam) == "coordinate":  # the extractor needs mates adjacent
                src = work / f"{sid}.dedup.namesorted.bam"
                _run(f"samtools sort -n -@ {args.threads} -m 1G -o {src} {bam}", log)
            _run(f"bismark_methylation_extractor -p --mbias_only --include_overlap --multicore {args.threads} "
                 f"-o {work} {src}", log)
            txt = work / (src.name[:-4] + ".M-bias.txt")
            shutil.copy(txt, dest)
            if src != bam:
                src.unlink()
        print(f"{sid}: {time.time() - t0:.0f} s -> {dest}", flush=True)


def scan_bam(bam: Path, contigs: list[str], min_mapq: int = 20, require_proper_pair: bool = True,
             max_pairs: int = 0) -> tuple[ScanAccumulator, dict]:
    """One pass over a Bismark PE BAM with the extractor's read filters (bad flags, MAPQ, contigs, proper pair)."""
    import pysam

    acc = ScanAccumulator()
    allowed = set(contigs)
    stats = {"reads_total": 0, "reads_filtered": 0, "pairs_discordant_contig": 0}
    pending = {}
    with pysam.AlignmentFile(str(bam), "rb", check_sq=False) as fh:
        for read in fh.fetch(until_eof=True):
            stats["reads_total"] += 1
            if (read.flag & _BAD_FLAGS or read.mapping_quality < min_mapq or read.reference_name not in allowed
                    or not read.is_paired or (require_proper_pair and not read.is_proper_pair)):
                stats["reads_filtered"] += 1; continue
            calls, L = read_cpg_calls(read.get_tag("XM"), read.get_tag("XG"), read.reference_start,
                                      read.cigartuples, read.is_reverse)
            five = read.reference_end if read.is_reverse else read.reference_start + 1  # 1-based 5'-most base
            rec = (read.reference_name, read.is_read1, calls, L, five)
            mate = pending.pop(read.query_name, None)
            if mate is None:
                pending[read.query_name] = rec; continue
            if mate[0] != rec[0]:
                stats["pairs_discordant_contig"] += 1; continue
            a, b = (rec, mate) if rec[1] else (mate, rec)
            acc.add_pair(a[2], a[3], a[4], b[2], b[3], b[4])
            if max_pairs and acc.pairs >= max_pairs:
                break
    stats["orphans"] = len(pending)
    return acc, stats


def _scan_one(sid: str, bam: str, out: str, contigs, min_mapq, proper, max_pairs) -> str:
    t0 = time.time()
    acc, stats = scan_bam(Path(bam), contigs, min_mapq, proper, max_pairs)
    acc.save(Path(out), {"sample_id": sid, "bam": bam, **stats, "seconds": round(time.time() - t0, 1)})
    return f"{sid}: {acc.pairs} pairs, {time.time() - t0:.0f} s"


def cmd_scan(cfg, args) -> None:
    outdir = cfg["mbias_dir"] / "scan"; outdir.mkdir(parents=True, exist_ok=True)
    pp = cfg["preprocessing"]
    jobs = [(s, str(_bam(cfg, s)), str(outdir / f"{s}.npz")) for s in args.samples
            if args.force or not (outdir / f"{s}.npz").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_scan_one, s, b, o, cfg["contigs"], pp["min_mapq"], pp["require_proper_pair"],
                            args.max_pairs) for s, b, o in jobs]
        for f in futs:
            print(f.result(), flush=True)


def cmd_validate(cfg, args) -> None:
    """Actually re-trim with another clip_r1 (under audits/work/validate_clip<k>/) and compare with the simulation."""
    from cfdna_origin.data.bam_fragments import extract_fragments
    from cfdna_origin.data.datasets import gse149438 as ds

    sid = args.sample
    row = _meta(cfg).set_index("sample_id").loc[sid].copy(); row["sample_id"] = sid
    pp = copy.deepcopy(cfg["preprocessing"]); pp["trim"]["clip_r1"] = args.clip_r1
    work = cfg["audit_dir"] / "work" / f"validate_clip{args.clip_r1}"
    bam = work / sid / f"{sid}.dedup.bam"
    (work / sid).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not bam.exists():
        with open(work / sid / "pipeline.log", "a") as log:
            for c in ds.sample_commands(row, raw_dir=work, bismark_index=cfg["reference_dir"], pp=pp,
                                        threads=args.threads):
                _run(c, log)
        for p in ds.intermediate_files(work, sid):
            p.unlink(missing_ok=True)
    t_pipe = time.time() - t0
    out = cfg["mbias_dir"] / "scan" / f"{sid}.clip_r1_{args.clip_r1}.npz"
    if not out.exists():
        _scan_one(sid, str(bam), str(out), cfg["contigs"], pp["min_mapq"], pp["require_proper_pair"], 0)
    off, keys, state, st = extract_fragments(bam, contigs=cfg["contigs"], min_mapq=pp["min_mapq"],
                                             require_proper_pair=pp["require_proper_pair"])
    n = np.diff(off)
    real = {"calls": int(len(keys)), "pct_meth": float(100 * state.mean()), "fragments_ge1": int(len(n)),
            **{f"fragments_ge{k}": int((n >= k).sum()) for k in range(2, KMAX + 1)}}
    base = np.load(cfg["mbias_dir"] / "scan" / f"{sid}.npz")
    sim = scenario_frame(base["sc"], base["scen"])
    s = sim[(sim.clip_r1 == args.clip_r1) & (sim.clip_r2 == pp["trim"]["clip_r2"]) & (sim.extra3 == 0)].iloc[0]
    b0 = sim[(sim.extra5_r1 == 0) & (sim.extra5_r2 == 0) & (sim.extra3 == 0)].iloc[0]
    simd = {"calls": int(s.calls), "pct_meth": float(s.pct_meth),
            **{f"fragments_ge{k}": int(s[f"frags_ge{k}"]) for k in range(1, KMAX + 1)}}
    b0d = {"calls": int(b0.calls), "pct_meth": float(b0.pct_meth),
           **{f"fragments_ge{k}": int(b0[f"frags_ge{k}"]) for k in range(1, KMAX + 1)}}
    res = {"sample_id": sid, "clip_r1": args.clip_r1, "pipeline_seconds": round(t_pipe), "bam_bytes": bam.stat().st_size,
           "baseline_clip10_from_scan": b0d, "simulated": simd, "observed_rerun": real,
           "observed_extract_stats": st,
           "rel_error": {k: (simd[k] - real[k]) / real[k] for k in simd if k in real and k != "pct_meth"},
           "pct_meth_error": simd["pct_meth"] - real["pct_meth"],
           "note": "observed calls come from extract_fragments (no reference-CpG filter); simulated from the scan of "
                   "the clip-10 BAM with the same read filters"}
    dest = cfg["audit_dir"] / "mbias_validation.json"
    dest.write_text(json.dumps(res, indent=2, default=int))
    print(json.dumps(res, indent=2, default=int))
    if args.delete_bam:
        bam.unlink()


def _profile_from_npz(z) -> tuple[pd.DataFrame, pd.DataFrame]:
    p5 = mbias_frame(z["m5"], end="5p")
    p3 = mbias_frame(z["m3"], end="3p")
    return p5, p3


def cmd_report(cfg, args) -> None:
    from scipy import stats as ss

    meta = _meta(cfg).set_index("sample_id")
    scan_dir, bis_dir, out = cfg["mbias_dir"] / "scan", cfg["mbias_dir"] / "bismark", cfg["audit_dir"]
    sids = sorted(p.stem for p in scan_dir.glob("GSM*.npz") if "." not in p.stem)
    if not sids:
        raise SystemExit("no scans")
    prof_rows, metric_rows, scen_rows, fe_rows, val_rows = [], [], [], [], []
    for sid in sids:
        z = np.load(scan_dir / f"{sid}.npz")
        info = {"sample_id": sid, "diagnosis": meta.loc[sid, "diagnosis"], "batch": meta.loc[sid, "batch"]}
        p5, p3 = _profile_from_npz(z)
        prof = pd.concat([p5, p3]); prof = prof.assign(**info); prof_rows.append(prof)
        total = int(p5.coverage.sum())
        for r, base, opts in ((1, 10, CLIP_R1), (2, 15, CLIP_R2 + (25,))):
            cm = clip_metrics(p5[p5.read == r], base, opts, total_calls=total, min_cov=args.min_cov)
            metric_rows.append(cm.assign(read=r, **info))
        sc = scenario_frame(z["sc"], z["scen"]).assign(**info); scen_rows.append(sc)
        fe = z["fe"]
        for end in (0, 1):
            for ri in (0, 1):
                cov = fe[end, ri, 0] + fe[end, ri, 1]
                d = np.flatnonzero(cov)
                fe_rows.append(pd.DataFrame({"end": ["R1_5p", "R2_5p"][end], "source_read": ri + 1, "dist": d,
                                             "meth": fe[end, ri, 1][d], "coverage": cov[d]}).assign(**info))
        bt = bis_dir / f"{sid}.M-bias.txt"
        if bt.exists():  # cross-check our 5' profile against Bismark (no MAPQ/contig/proper-pair filter there)
            b = parse_mbias_txt(bt); b = b[b.context == "CpG"]
            mg = b.merge(p5, on=["read", "position"], suffixes=("_bis", "_scan"))
            mg = mg[mg.coverage_bis >= args.min_cov]
            val_rows.append({**info, "max_abs_pct_diff": float((mg.pct_bis - mg.pct_scan).abs().max()),
                             "coverage_ratio_scan_over_bismark": float(mg.coverage_scan.sum() / mg.coverage_bis.sum())})
    prof = pd.concat(prof_rows); metrics = pd.concat(metric_rows); scen = pd.concat(scen_rows)
    fe = pd.concat(fe_rows)
    fe["pct"] = 100 * fe.meth / fe.coverage
    prof.to_csv(out / "mbias_profiles.csv", index=False)
    metrics.to_csv(out / "mbias_clip_metrics.csv", index=False)
    scen.to_csv(out / "mbias_scenarios.csv", index=False)
    fe.to_csv(out / "mbias_fragment_end_profiles.csv", index=False)
    if val_rows:
        pd.DataFrame(val_rows).to_csv(out / "mbias_bismark_crosscheck.csv", index=False)

    # group summaries: mean (min-max) of per-sample metrics, per batch / per class / all
    cols = ["calls_lost_frac_all", "delta_pct_meth_read", "resid_abs_dev", "resid_signed_dev", "first_cycle_dev",
            "plateau"]
    grp = []
    for key in (None, "batch", "diagnosis"):
        g = metrics.groupby(["read", "clip"] + ([key] if key else []))[cols].mean().reset_index()
        g["group_by"] = key or "all"; g["group"] = g[key] if key else "all"
        grp.append(g.drop(columns=[key]) if key else g)
    pd.concat(grp).to_csv(out / "mbias_clip_metrics_by_group.csv", index=False)
    sc_cols = ["calls_lost_frac", "delta_pct_meth", "frags_ge1_lost_frac", "frags_ge3_lost_frac", "conflicts"]
    sgrp = []
    for key in (None, "batch", "diagnosis"):
        g = scen.groupby(["clip_r1", "clip_r2", "extra3"] + ([key] if key else []))[sc_cols].mean().reset_index()
        g["group_by"] = key or "all"; g["group"] = g[key] if key else "all"
        sgrp.append(g.drop(columns=[key]) if key else g)
    pd.concat(sgrp).to_csv(out / "mbias_scenarios_by_group.csv", index=False)

    # batch test of the M-bias SHAPE: per-sample signed deviation of the first retained cycles (current trimming)
    shape = {}
    for r, c in ((1, 10), (1, 15), (1, 20), (2, 15), (2, 20)):
        m = metrics[(metrics.read == r) & (metrics.clip == c)]
        a, b = m[m.batch == "KRp1"].resid_signed_dev, m[m.batch == "KRp2"].resid_signed_dev
        p = float(ss.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan
        shape[f"R{r}_clip{c}"] = {"KRp1_mean": float(a.mean()), "KRp2_mean": float(b.mean()), "mannwhitney_p": p,
                                  "n": [len(a), len(b)]}
    # 3' ends: signed deviation of the last 10 retained positions from the read's plateau
    tail = []
    for sid, g in prof[prof.end == "3p"].groupby("sample_id"):
        for r in (1, 2):
            gp = g[g.read == r].sort_values("position")
            pl = metrics[(metrics.sample_id == sid) & (metrics.read == r)].plateau.iloc[0]
            gp = gp[gp.coverage >= args.min_cov]
            tail.append({"sample_id": sid, "batch": gp.batch.iloc[0], "diagnosis": gp.diagnosis.iloc[0], "read": r,
                         "last10_signed_dev": float((gp.head(10).pct - pl).mean()),
                         "last5_signed_dev": float((gp.head(5).pct - pl).mean()),
                         "last10_calls_frac": float(gp.head(10).coverage.sum() / gp.coverage.sum())})
    tail = pd.DataFrame(tail); tail.to_csv(out / "mbias_3prime.csv", index=False)
    # cycle vs molecule: excess at the first 10 bp of each fragment end, measured with each mate's calls
    endx = fragment_end_excess(fe)
    endx.to_csv(out / "mbias_fragment_end_excess.csv", index=False)
    # non-CpG (conversion) M-bias from Bismark: excess CHH/CHG at the first 5 retained positions
    nc = []
    for sid in sids:
        bt = bis_dir / f"{sid}.M-bias.txt"
        if bt.exists():
            nc.append({"sample_id": sid, "batch": meta.loc[sid, "batch"], "diagnosis": meta.loc[sid, "diagnosis"],
                       **noncpg_excess(parse_mbias_txt(bt))})
    nc = pd.DataFrame(nc)
    if len(nc):
        nc.to_csv(out / "mbias_noncpg_excess.csv", index=False)
    shape_extra = {}
    for col in [c for c in endx.columns if c.startswith("excess_")]:
        a, b = endx[endx.batch == "KRp1"][col], endx[endx.batch == "KRp2"][col]
        shape_extra[col] = {"KRp1_mean": float(a.mean()), "KRp2_mean": float(b.mean()),
                            "mannwhitney_p": float(ss.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan}
    summary = {"samples": sids, "batch_shape_test": shape, "batch_fragment_end_test": shape_extra,
               "fragment_end_excess_mean": endx.drop(columns=["sample_id", "batch", "diagnosis"]).mean().round(3)
               .to_dict(),
               "noncpg_excess_mean": nc.drop(columns=["sample_id", "batch", "diagnosis"]).mean().round(3).to_dict()
               if len(nc) else {},
               "clip_metrics_all": pd.concat(grp).query("group_by == 'all'").round(4).to_dict("records"),
               "scenarios_all": pd.concat(sgrp).query("group_by == 'all'").round(5).to_dict("records"),
               "three_prime": tail.groupby("read")[["last10_signed_dev", "last5_signed_dev",
                                                    "last10_calls_frac"]].mean().round(4).to_dict("index"),
               "bismark_crosscheck": val_rows}
    (out / "mbias_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("batch_shape_test", "three_prime")}, indent=2, default=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("bismark"); b.add_argument("--samples", nargs="+", required=True)
    b.add_argument("--threads", type=int, default=4); b.add_argument("--force", action="store_true")
    s = sub.add_parser("scan"); s.add_argument("--samples", nargs="+", required=True)
    s.add_argument("--workers", type=int, default=4); s.add_argument("--force", action="store_true")
    s.add_argument("--max-pairs", type=int, default=0, help="debug: stop after N pairs")
    v = sub.add_parser("validate"); v.add_argument("--sample", required=True)
    v.add_argument("--clip-r1", type=int, default=20); v.add_argument("--threads", type=int, default=16)
    v.add_argument("--delete-bam", action="store_true")
    r = sub.add_parser("report"); r.add_argument("--min-cov", type=int, default=1000,
                                                 help="ignore read positions with fewer calls")
    args = ap.parse_args()
    cfg = _cfg()
    {"bismark": cmd_bismark, "scan": cmd_scan, "validate": cmd_validate, "report": cmd_report}[args.step](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
