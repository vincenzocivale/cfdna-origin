import re

import numpy as np
import pytest

from cfdna_origin.data.bam_fragments import extract_fragments
from cfdna_origin.data.loci import make_keys

pysam = pytest.importorskip("pysam")


def _xm(length, calls):
    s = ["."] * length
    for i, c in calls.items():
        s[i] = c
    return "".join(s)


def _read(header, name, ref, start, flag, calls, *, cigar="20M", xg="CT", mapq=40, mate_start=None):
    a = pysam.AlignedSegment(header)
    qlen = sum(int(n) for n, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar) if op in "MIS=X")
    a.query_name, a.flag, a.reference_name, a.reference_start = name, flag, ref, start
    a.cigarstring, a.query_sequence, a.mapping_quality = cigar, "A" * qlen, mapq
    a.query_qualities = pysam.qualitystring_to_array("I" * qlen)
    if mate_start is not None:
        a.next_reference_name, a.next_reference_start = ref, mate_start
    a.set_tag("XM", _xm(qlen, calls), "Z")
    a.set_tag("XG", xg, "Z")
    return a


def _write_bam(path, contigs, reads_fn):
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": c, "LN": 1000} for c in contigs]})
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for r in reads_fn(header, contigs[0], contigs[1]):
            out.write(r)
    return path


def _reads(h, main, other):
    R1, R2 = 0x1 | 0x2 | 0x40 | 0x20, 0x1 | 0x2 | 0x80 | 0x10
    return [
        # pair 1, top strand, mates overlap on ref 110 with concordant 'z'
        _read(h, "p1", main, 100, R1, {2: "Z", 10: "z"}, mate_start=105),
        _read(h, "p1", main, 105, R2, {5: "z", 15: "Z"}, mate_start=100),
        # pair 2: ref 208 called Z by one mate and z by the other -> dropped
        _read(h, "p2", main, 200, R1, {4: "Z", 8: "Z"}, mate_start=205),
        _read(h, "p2", main, 205, R2, {3: "z", 10: "Z"}, mate_start=200),
        # single-end bottom-strand read with an insertion (the call inside the insertion has no ref position)
        _read(h, "ga", main, 300, 0x10, {1: "Z", 4: "z", 6: "z"}, cigar="3M2I10M", xg="GA"),
        _read(h, "lowq", main, 400, 0, {1: "Z"}, mapq=5),
        _read(h, "dup", main, 500, 0x400, {1: "Z"}),
        _read(h, "other", other, 100, 0, {1: "Z"}),
    ]


def test_extract_fragments(tmp_path):
    bam = _write_bam(tmp_path / "toy.bam", ["chr1", "chr2"], _reads)
    off, keys, state, stats = extract_fragments(bam, contigs=["chr1"], min_mapq=20)
    np.testing.assert_array_equal(off, [0, 3, 5, 7])
    np.testing.assert_array_equal(keys, make_keys("chr1", [103, 111, 121, 205, 216, 301, 304]))
    np.testing.assert_array_equal(state, [1, 0, 1, 1, 1, 1, 0])
    assert state.dtype == np.uint8 and keys.dtype == np.int64
    assert stats["reads_total"] == 8 and stats["fragments_kept"] == 3
    assert stats["cpg_mate_conflicts_dropped"] == 1
    assert stats["reads_bad_flag"] == 1 and stats["reads_low_mapq"] == 1 and stats["reads_other_contig"] == 1
    assert stats["reads_orphan_mate"] == 0


def test_extract_fragments_orphan_and_improper(tmp_path):
    def reads(h, main, other):
        return [_read(h, "a", main, 100, 0x1 | 0x2 | 0x40, {2: "Z"}, mate_start=150),  # mate missing
                _read(h, "b", main, 200, 0x1 | 0x40, {2: "Z"}, mate_start=250)]         # not a proper pair
    bam = _write_bam(tmp_path / "o.bam", ["chr1", "chr2"], reads)
    off, keys, _, stats = extract_fragments(bam, contigs=["chr1"])
    assert len(off) == 1 and len(keys) == 0
    assert stats["reads_orphan_mate"] == 1 and stats["reads_not_proper_pair"] == 1


def test_extract_fragments_ensembl_contig_names(tmp_path):
    bam = _write_bam(tmp_path / "ens.bam", ["1", "2"], _reads)
    _, keys, _, _ = extract_fragments(bam, contigs=["1"])
    np.testing.assert_array_equal(keys[:3], make_keys("chr1", [103, 111, 121]))
