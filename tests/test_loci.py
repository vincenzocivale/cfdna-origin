import numpy as np
import pytest

from cfdna_origin.data.loci import (
    CODE_CHROM,
    GRCH38_LENGTHS,
    key_chrom_code,
    key_position,
    make_keys,
    normalize_chrom,
    random_genome_keys,
)


def test_make_keys_roundtrip():
    pos = np.array([1, 10_000, 248_956_422])
    keys = make_keys("chr1", pos)
    assert keys.dtype == np.int64
    np.testing.assert_array_equal(key_position(keys), pos)
    np.testing.assert_array_equal(key_chrom_code(keys), 1)
    mixed = make_keys(np.array(["chr2", "22", "chrX", "Y"]), np.array([5, 6, 7, 8]))
    np.testing.assert_array_equal(key_chrom_code(mixed), [2, 22, 23, 24])
    np.testing.assert_array_equal(key_position(mixed), [5, 6, 7, 8])


def test_sex_chromosome_codes():
    assert key_chrom_code(make_keys("chrX", [100]))[0] == 23
    assert key_chrom_code(make_keys("chrY", [100]))[0] == 24
    assert CODE_CHROM[23] == "chrX" and CODE_CHROM[24] == "chrY"
    assert normalize_chrom("X") == "chrX" and normalize_chrom("chr7") == "chr7"


@pytest.mark.parametrize("bad", [0, -5, 2**32])
def test_invalid_positions_raise(bad):
    with pytest.raises(ValueError):
        make_keys("chr1", [1, bad])


def test_unknown_chromosome_raises():
    with pytest.raises(KeyError):
        make_keys("chrM", [1])


def test_random_genome_keys_deterministic_and_in_range():
    a, b = random_genome_keys(2000, seed=3), random_genome_keys(2000, seed=3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, random_genome_keys(2000, seed=4))
    assert np.all(np.diff(a) > 0)  # sorted unique
    codes, pos = key_chrom_code(a), key_position(a)
    assert set(np.unique(codes)) <= set(range(1, 23))  # autosomes by default
    lengths = np.array([GRCH38_LENGTHS[CODE_CHROM[int(c)]] for c in codes])
    assert np.all(pos >= 1) and np.all(pos <= lengths)
    only_x = random_genome_keys(100, seed=0, chroms=["chrX"])
    assert set(key_chrom_code(only_x)) == {23}
