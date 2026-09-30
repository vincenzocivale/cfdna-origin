import gzip

import numpy as np
import pytest

from cfdna_origin.data.liftover import Chain
from cfdna_origin.data.loci import make_keys
from cfdna_origin.data.reference import cpg_positions, read_fasta, reference_cpg_keys

# chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
CHAIN = """\
chain 1000 chr1 10000 + 100 300 chr1 20000 + 1000 1210 1
50 20 30
130

chain 900 chr2 1000 + 0 100 chr3 5000 - 200 300 2
100

chain 800 chr1 10000 + 400 600 chr1 20000 + 2000 2200 3
200

chain 700 chr1 10000 + 450 460 chr5 20000 + 10 20 4
10

chain 600 chr1_gl000191_random 5000 + 0 50 chr1 20000 + 0 50 5
50
"""


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    path = tmp_path_factory.mktemp("chain") / "toy.over.chain.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(CHAIN)
    return Chain(path)


def test_plus_strand_blocks_and_gap(chain):
    out, ok, stats = chain.lift(make_keys("chr1", [101, 150, 160, 171, 300, 50]))
    np.testing.assert_array_equal(ok, [True, True, False, True, True, False])
    np.testing.assert_array_equal(out[ok], make_keys("chr1", [1001, 1050, 1081, 1210]))
    assert stats["unmapped"] == 2 and stats["ambiguous"] == 0 and stats["kept"] == 4 and stats["n"] == 6


def test_minus_strand_maps_c_to_target_c(chain):
    # source CpG C=11,G=12 (1-based) on chr2; block t[0,100) -> reverse complement of chr3 fwd [4700, 4800):
    # C(11) lands on 4790 and G(12) on 4789, so the target CpG's C is 4789 (= mapped base - 1)
    out, ok, _ = chain.lift(make_keys("chr2", [11, 1]))
    assert ok.all()
    np.testing.assert_array_equal(out, make_keys("chr3", [4789, 4799]))


def test_overlapping_chains_ambiguous_and_nested(chain):
    out, ok, stats = chain.lift(make_keys("chr1", [401, 455, 501, 600, 601]))
    np.testing.assert_array_equal(ok, [True, False, True, True, False])
    np.testing.assert_array_equal(out[ok], make_keys("chr1", [2001, 2101, 2200]))  # 501: covered by chain 3 only
    assert stats["ambiguous"] == 1 and stats["unmapped"] == 1


def test_non_primary_contig_chains_ignored(chain):
    assert set(chain.b[:, 0]) == {1, 2}  # the chr1_gl000191_random chain is skipped


def test_target_cpg_filter(chain):
    keys = make_keys("chr1", [101, 150, 171, 160])
    out, ok, stats = chain.lift(keys, target_cpg_keys=make_keys("chr1", [1001, 1081]))
    np.testing.assert_array_equal(ok, [True, False, True, False])
    assert stats["not_cpg_in_target"] == 1 and stats["unmapped"] == 1 and stats["kept"] == 2


def test_cpg_positions():
    np.testing.assert_array_equal(cpg_positions(b"ACGTcgNNCGGC"), [2, 5, 9])
    assert len(cpg_positions(b"CCCC")) == 0


def test_read_fasta_and_reference_keys(tmp_path):
    fa = tmp_path / "toy.fa.gz"
    with gzip.open(fa, "wb") as fh:
        fh.write(b">chr1 description\nACGT\nACG\n>chrUn\nCGCG\n>chr2\nTTCG\n")
    seqs = dict(read_fasta(fa, {"chr1", "chr2"}))
    assert seqs == {"chr1": b"ACGTACG", "chr2": b"TTCG"}
    cache = tmp_path / "cache" / "cpgs.npy"
    keys = reference_cpg_keys(fa, cache, contigs=["chr1", "chr2"])
    np.testing.assert_array_equal(keys, np.sort(np.concatenate([make_keys("chr1", [2, 6]), make_keys("chr2", [3])])))
    assert cache.exists()
    np.testing.assert_array_equal(reference_cpg_keys(fa, cache, contigs=["chr1", "chr2"]), keys)
    with pytest.raises(ValueError, match="lacks"):
        reference_cpg_keys(fa, tmp_path / "other.npy", contigs=["chr1", "chr3"])
