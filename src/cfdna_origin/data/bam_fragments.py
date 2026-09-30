"""Bismark BAM -> per-fragment CpG methylation patterns.

Bismark writes a per-read methylation call string in the `XM` tag (Z = methylated CpG, z = unmethylated CpG, aligned
to the read sequence) and the genomic strand in `XG` (CT = top strand: the call is on the C; GA = bottom strand: the
call is on the G of the CpG, i.e. C position = G position - 1). Both datasets' published pipelines use Bismark
(HRA003209) or can be re-run with it (GSE149438), so a single parser serves both.

A fragment = one read pair (or one single-end read). Calls from overlapping mates are merged; a CpG called
discordantly by the two mates is dropped. Reads are filtered (MAPQ, proper pair, not secondary/supplementary/QC-fail/
duplicate) and restricted to the configured contigs. Every filter is counted in the returned stats.
"""
from __future__ import annotations

import re
from array import array
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np

from cfdna_origin.data.loci import CHROM_CODE, normalize_chrom

_BAD_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800  # unmapped, secondary, QC fail, duplicate, supplementary
_CALL = re.compile("[Zz]")
_QREF, _QONLY, _RONLY = (0, 7, 8), (1, 4), (2, 3)  # CIGAR ops consuming query+ref (M,=,X), query (I,S), ref (D,N)


def read_calls(read) -> tuple[list[int], list[int]]:
    """(1-based C positions, states) of the CpG calls of one aligned read. Plain lists: per-read numpy is ~3x slower."""
    xm = read.get_tag("XM")
    hits = [m.start() for m in _CALL.finditer(xm)]
    if not hits:
        return [], []
    base = read.reference_start + (1 if read.get_tag("XG") == "CT" else 0)  # 0-based ref -> 1-based C of the CpG
    cig = read.cigartuples
    if len(cig) == 1 and cig[0][0] in _QREF:  # ungapped (the vast majority)
        return [base + h for h in hits], [int(xm[h] == "Z") for h in hits]
    qs, rs, ls = [], [], []  # aligned blocks (query start incl. soft clips, ref offset, length)
    q = r = 0
    for op, n in cig:
        if op in _QREF:
            qs.append(q); rs.append(r); ls.append(n); q += n; r += n
        elif op in _QONLY:
            q += n
        elif op in _RONLY:
            r += n
    pos, st = [], []
    for h in hits:  # calls inside insertions / soft clips have no reference position
        i = bisect_right(qs, h) - 1
        if i >= 0 and h < qs[i] + ls[i]:
            pos.append(base + rs[i] + h - qs[i]); st.append(int(xm[h] == "Z"))
    return pos, st


def _merge(pos_a, st_a, pos_b, st_b):
    """Union of two mates' calls, position-sorted; a CpG called discordantly by the two mates is dropped."""
    calls = dict(zip(pos_a, st_a))
    conflict = set()
    for p, s in zip(pos_b, st_b):
        if calls.setdefault(p, s) != s:
            conflict.add(p)
    for p in conflict:
        del calls[p]
    pos = sorted(calls)
    return pos, [calls[p] for p in pos], len(conflict)


def extract_fragments(bam_path: Path, *, contigs: list[str], min_mapq: int = 20, paired: bool = True,
                      require_proper_pair: bool = True, max_buffer: int = 5_000_000):
    """Returns (frag_offsets, locus_key, state, stats). Fragments with no CpG call are not stored."""
    import pysam

    allowed = set(contigs)
    stats = Counter()
    offsets, keys, states = array("q", [0]), array("q"), array("B")
    pending: dict[str, tuple] = {}

    def emit(chrom, pos, st):
        if not pos:
            stats["fragments_without_cpg"] += 1
            return
        code = CHROM_CODE[normalize_chrom(chrom)] << 32
        keys.extend([code | p for p in pos]); states.extend(st)
        offsets.append(len(keys))
        stats["fragments_kept"] += 1

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            stats["reads_total"] += 1
            if read.flag & _BAD_FLAGS:
                stats["reads_bad_flag"] += 1; continue
            if read.mapping_quality < min_mapq:
                stats["reads_low_mapq"] += 1; continue
            chrom = read.reference_name
            if chrom not in allowed:
                stats["reads_other_contig"] += 1; continue
            if paired and read.is_paired:
                if require_proper_pair and not read.is_proper_pair:
                    stats["reads_not_proper_pair"] += 1; continue
                mate = pending.pop(read.query_name, None)
                calls = read_calls(read)
                if mate is None:
                    pending[read.query_name] = (chrom, *calls)
                    if len(pending) > max_buffer:
                        raise RuntimeError("mate buffer overflow: BAM is neither name-grouped nor coordinate-sorted pairs")
                    continue
                if mate[0] != chrom:
                    stats["pairs_discordant_contig"] += 1; continue
                pos, st, n_conflict = _merge(mate[1], mate[2], *calls)
                stats["cpg_mate_conflicts_dropped"] += n_conflict
                emit(chrom, pos, st)
            else:
                pos, st = read_calls(read)
                order = sorted(range(len(pos)), key=pos.__getitem__)
                emit(chrom, [pos[i] for i in order], [st[i] for i in order])
    stats["reads_orphan_mate"] = len(pending)
    return (np.frombuffer(offsets, np.int64).copy(), np.frombuffer(keys, np.int64).copy(),
            np.frombuffer(states, np.uint8).copy(), dict(stats))
