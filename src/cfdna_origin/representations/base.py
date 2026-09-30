"""The LocusEmbeddingStore interface: frozen, patient-agnostic CpG locus representations consumed as artifacts.

A store maps GRCh38 locus keys (see `cfdna_origin.data.loci`) to fixed vectors. Stores are never trained here.
Every store carries a `RepresentationManifest` with the provenance fields that must be recorded with each run.
Lookups of loci outside the store's universe raise `RepresentationCoverageError`; they are never silently dropped.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cfdna_origin.data.loci import GENOME_BUILD, KEY_NAMESPACE, make_keys


class RepresentationCoverageError(ValueError):
    pass


@dataclasses.dataclass
class RepresentationManifest:
    name: str
    kind: str  # hdf5 | random | position | none
    dim: int
    source: str  # artifact path / checkpoint / generator spec
    genome_build: str = GENOME_BUILD
    key_namespace: str = KEY_NAMESPACE
    cpg_universe: str = ""  # human-readable description of the covered loci
    n_universe_loci: int | None = None
    preprocessing: dict[str, Any] = dataclasses.field(default_factory=dict)
    normalization: dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)  # hashes, attrs, created time

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class LocusEmbeddingStore(ABC):
    """Frozen lookup from locus keys to embeddings (float32 [n, dim])."""

    manifest: RepresentationManifest

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def dim(self) -> int:
        return self.manifest.dim

    @abstractmethod
    def covers(self, keys: np.ndarray) -> np.ndarray:
        """Boolean mask: which keys are inside the store's universe."""

    @abstractmethod
    def _lookup(self, keys: np.ndarray) -> np.ndarray:
        """Embeddings for keys already known to be covered (any order)."""

    def get_embeddings(self, keys: np.ndarray) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.int64)
        covered = self.covers(keys)
        if not covered.all():
            missing = keys[~covered]
            raise RepresentationCoverageError(
                f"{self.name}: {len(missing)}/{len(keys)} loci outside the representation universe "
                f"({self.manifest.cpg_universe}); first missing keys: {missing[:5].tolist()}"
            )
        return self._lookup(keys)

    def reference_sample(self, n: int, seed: int) -> np.ndarray:
        """Label-free embeddings of a seeded genome-wide locus sample (for fitting the common PCA projection)."""
        from cfdna_origin.data.loci import random_genome_keys

        return self._lookup(random_genome_keys(n, seed))

    def get_embedding(self, chrom: str, position: int) -> np.ndarray:
        return self.get_embeddings(make_keys(chrom, np.asarray([position])))[0]


class NoLocusStore(LocusEmbeddingStore):
    """`methylation_only`: no locus information at all (dim 0). The model uses a learned constant token."""

    def __init__(self, name: str = "methylation_only"):
        self.manifest = RepresentationManifest(name=name, kind="none", dim=0, source="none",
                                               cpg_universe="any locus (no locus information used)",
                                               normalization={"type": "none", "data_derived": False})

    def covers(self, keys: np.ndarray) -> np.ndarray:
        return np.ones(len(keys), dtype=bool)

    def _lookup(self, keys: np.ndarray) -> np.ndarray:
        return np.zeros((len(keys), 0), dtype=np.float32)
