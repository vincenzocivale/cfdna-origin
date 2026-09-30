"""Unit tests of the pure helpers of scripts/audit_mbias.py and scripts/audit_fragment_storage.py (synthetic, fast)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mb = _load("audit_mbias")
fs = _load("audit_fragment_storage")

MBIAS_TXT = """CpG context (R1)
================
position\tcount methylated\tcount unmethylated\t% methylation\tcoverage
1\t8\t2\t80.00\t10
2\t6\t4\t60.00\t10

CHG context (R1)
================
position\tcount methylated\tcount unmethylated\t% methylation\tcoverage
1\t0\t10\t0.00\t10

CpG context (R2)
================
position\tcount methylated\tcount unmethylated\t% methylation\tcoverage
1\t5\t5\t50.00\t10
2\t0\t0\t\t0
"""


def test_parse_mbias_txt(tmp_path):
    p = tmp_path / "x.M-bias.txt"; p.write_text(MBIAS_TXT)
    df = mb.parse_mbias_txt(p)
    assert len(df) == 5
    r1 = df[(df.context == "CpG") & (df.read == 1)]
    assert r1.meth.tolist() == [8, 6] and r1.coverage.tolist() == [10, 10]
    assert df[(df.context == "CHG")].unmeth.item() == 10
    assert np.isnan(df[(df.read == 2) & (df.position == 2)].pct.item())
    assert mb.parse_mbias_txt(MBIAS_TXT).equals(df)  # text input


def test_read_cpg_calls_forward_reverse_and_indel():
    xm = "..Z...z..."  # hits at query 2 (meth) and 6 (unmeth)
    calls, L = mb.read_cpg_calls(xm, "CT", 100, [(0, 10)], False)
    assert L == 10 and calls == [(103, 1, 2), (107, 0, 6)]  # 0-based ref 100 -> 1-based C 101 + h
    calls, _ = mb.read_cpg_calls(xm, "GA", 100, [(0, 10)], True)  # bottom strand: C = G - 1; 5' end is XM start
    assert calls == [(102, 1, 7), (106, 0, 3)]
    # 2M 2I 6M: query 2,3 are insertions -> the call at query 2 is dropped; query 6 -> ref offset 4
    calls, _ = mb.read_cpg_calls(xm, "CT", 100, [(0, 2), (1, 2), (0, 6)], False)
    assert calls == [(105, 0, 6)]
    # 3M 2D 7M: query 6 -> ref offset 8
    calls, _ = mb.read_cpg_calls(xm, "CT", 100, [(0, 3), (2, 2), (0, 7)], False)
    assert calls == [(103, 1, 2), (109, 0, 6)]
    assert mb.read_cpg_calls("....", "CT", 0, [(0, 4)], False) == ([], 4)


def test_read_cpg_calls_matches_extractor():
    from cfdna_origin.data import bam_fragments as bf

    class R:  # minimal pysam-like read
        def __init__(self, xm, xg, start, cig):
            self._t = {"XM": xm, "XG": xg}; self.reference_start = start; self.cigartuples = cig

        def get_tag(self, k):
            return self._t[k]

    for args in (("..Z...z..Z", "CT", 50, [(0, 10)]), ("Z.z..Z....", "GA", 7, [(0, 3), (1, 1), (0, 4), (2, 3), (0, 2)])):
        pos, st = bf.read_calls(R(*args))
        calls, _ = mb.read_cpg_calls(*args, False)
        assert [c[0] for c in calls] == pos and [c[1] for c in calls] == st


def test_merge_pair_clipping_and_conflicts():
    r1 = [(100, 1, 0), (110, 1, 5), (120, 0, 9)]  # (pos, state, pos5); read length 10
    r2 = [(120, 1, 0), (130, 0, 5)]
    assert mb.merge_pair(r1, 10, r2, 10, 0, 0, 0) == (3, 2, 1)  # 120 discordant -> dropped
    assert mb.merge_pair(r1, 10, r2, 10, 1, 0, 0) == (2, 1, 1)  # R1 pos5 0 clipped
    assert mb.merge_pair(r1, 10, r2, 10, 0, 1, 0) == (4, 2, 0)  # R2's 120 clipped -> no conflict
    assert mb.merge_pair(r1, 10, r2, 10, 0, 0, 1) == (4, 3, 0)  # 3' clip removes R1 pos5 9 (120): R2 call wins
    assert mb.merge_pair(r1, 10, r2, 10, 10, 10, 0) == (0, 0, 0)


def test_scan_accumulator_and_frames():
    acc = mb.ScanAccumulator()
    acc.add_pair([(100, 1, 0), (105, 0, 5)], 20, 100, [(105, 0, 3)], 20, 108)
    assert acc.pairs == 1
    assert acc.m5[0, 1, 0] == 1 and acc.m5[0, 0, 5] == 1 and acc.m5[1, 0, 3] == 1
    assert acc.m3[0, 1, 19] == 1 and acc.m3[1, 0, 16] == 1
    assert acc.fe[0, 0, 1, 0] == 1 and acc.fe[1, 1, 0, 3] == 1  # distances from R1-5' and R2-5' ends
    sc = mb.scenario_frame(acc.sc, np.asarray(acc.scen))
    base = sc[(sc.clip_r1 == 10) & (sc.clip_r2 == 15) & (sc.extra3 == 0)].iloc[0]
    assert base.calls == 2 and base.meth == 1 and base.frags_ge2 == 1
    c20 = sc[(sc.clip_r1 == 20) & (sc.clip_r2 == 15) & (sc.extra3 == 0)].iloc[0]
    assert c20.calls == 1 and c20.calls_lost_frac == 0.5 and c20.delta_pct_meth == -50
    prof = mb.mbias_frame(acc.m5)
    assert set(prof.read) == {1, 2} and prof[prof.read == 1].cycle.min() == 11 and prof[prof.read == 2].cycle.min() == 19


def test_clip_metrics():
    pos = np.arange(1, 121)
    pct = np.where(pos <= 10, 80.0, 60.0)  # first 10 R1 positions biased +20 pts
    cov = np.full(len(pos), 100)
    prof = pd.DataFrame({"position": pos, "cycle": pos + 10, "coverage": cov, "meth": (pct * cov / 100).astype(int),
                         "pct": pct})
    cm = mb.clip_metrics(prof, 10, (10, 15, 20), total_calls=24000).set_index("clip")
    assert cm.loc[10, "plateau"] == 60
    assert cm.loc[10, "resid_abs_dev"] == pytest.approx(20) and cm.loc[15, "resid_abs_dev"] == pytest.approx(10)
    assert cm.loc[20, "resid_abs_dev"] == 0 and cm.loc[20, "calls_lost_frac_read"] == pytest.approx(10 / 120)
    assert cm.loc[20, "calls_lost_frac_all"] == pytest.approx(1000 / 24000)
    assert cm.loc[20, "pct_meth_read"] == pytest.approx(60) and cm.loc[10, "delta_pct_meth_read"] == 0


def test_cpg_hist_and_threshold_table():
    n = np.array([1, 2, 3, 3, 45])
    h = fs.cpg_hist(n, kmax=40)
    assert len(h) == 40 and h[0] == 1 and h[1] == 1 and h[2] == 2 and h[-1] == 1
    off = np.concatenate([[0], np.cumsum([1, 2, 3, 4])])
    state = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1], np.uint8)  # frags: M | MM | UUU | MUMM
    tt = fs.threshold_table(off, state, (1, 3)).set_index("min_cpgs")
    assert tt.loc[1, "fragments"] == 4 and tt.loc[1, "calls"] == 10 and tt.loc[1, "frac_fully_meth"] == 0.5
    assert tt.loc[3, "fragments"] == 2 and tt.loc[3, "calls"] == 7 and tt.loc[3, "calls_frac"] == 0.7
    assert tt.loc[3, "frac_fully_meth"] == 0 and tt.loc[3, "frac_fully_unmeth"] == 0.5
    assert tt.loc[3, "pct_meth"] == pytest.approx(300 / 7) and tt.loc[3, "mean_cpgs_per_fragment"] == 3.5
    assert tt.loc[3, "bytes"] == 7 * 5 + 3 * 8
    with pytest.raises(ValueError):
        fs.threshold_table(np.array([0, 1, 1]), np.array([1], np.uint8))


def test_extrapolate():
    per = pd.DataFrame({"sample_id": ["a", "b"], "min_cpgs": [3, 3], "bytes": [100, 300], "read_pairs": [10, 10],
                        "batch": ["KRp1", "KRp2"]})
    meta = pd.DataFrame({"read_pairs": [10, 20, 30], "batch": ["KRp1", "KRp1", "KRp2"]})
    e = fs.extrapolate(per, meta).iloc[0]
    assert e.bytes_per_read_pair_median == 20 and e.total_GB_median == pytest.approx(60 * 20 / 1e9)
    assert e.total_GB_min == pytest.approx(600 / 1e9) and e.total_GB_max == pytest.approx(1800 / 1e9)
    assert e.total_GB_batch_median == pytest.approx((30 * 10 + 30 * 30) / 1e9)


def test_parse_dedup_report():
    txt = ("Total number of alignments analysed in /x/y.bam:\t1000\n"
           "Total number duplicated alignments removed:\t880 (88.00%)\n")
    assert fs.parse_dedup_report(txt) == {"aligned_pairs": 1000, "duplicates_removed": 880, "dup_rate": 0.88}
    assert fs.parse_dedup_report("garbage") == {}


def test_fragment_end_excess_and_noncpg():
    fe = pd.DataFrame({"sample_id": "s", "batch": "KRp1", "diagnosis": "Normal", "end": "R1_5p", "source_read": 1,
                       "dist": [0, 5, 50, 60], "meth": [8, 8, 6, 6], "coverage": [10, 10, 10, 10]})
    ex = mb.fragment_end_excess(fe)
    assert ex.excess_R1_5p_R1.item() == pytest.approx(20)
    b = mb.parse_mbias_txt(MBIAS_TXT)
    rows = [("CHH", r, p, 1, 99, 10.0 if p <= 5 else 1.0, 100) for r in (1, 2) for p in range(1, 90)]
    b = pd.DataFrame(rows, columns=b.columns)
    nc = mb.noncpg_excess(b)
    assert nc["CHH_R1_first5_excess"] == pytest.approx(9) and nc["CHH_R2_first5_excess"] == pytest.approx(9)
