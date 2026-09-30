# Repository and data audit (pre-refactor, commit 3919456)

Audit of the repository as it was before `MILESTONE_1_REFACTOR`, against the new objective
(real-cfDNA, sample-level tissue/cancer-of-origin classification; the CpG representation is the only experimental
variable). The pre-refactor tree is preserved in git as tag `legacy-loyfer-v0`.

## 0. What the old repository did

Loyfer et al. WGBS atlas (GSE186458, 253 purified cell-type samples) -> PAT files -> reads thinned 1% (>= 4 CpGs)
-> per-read transformer (CLS token) supervised with the **sample's cell-type label copied onto every read** ->
per-read accuracy, sample accuracy by mean log-prob, and "cfDNA composition" on 23 external healthy cfDNA samples by
`argmax per read -> bincount`. No training had been run beyond a 300-step smoke test (no results exist).

## 1. Code still useful

| File | Verdict | Why |
|---|---|---|
| `models/set_classifier.py` token construction (frozen locus embedding + state embedding + log-gap geometry) | idea reused, rewritten | Becomes the Level-1 fragment encoder token; the CLS transformer + per-read heads are replaced |
| `training/tables.py` checks (row order vs index, fail on mismatch) | idea reused | Coverage/order validation moved to `representations/` |
| `data/readstore.py` offset-based ragged read storage | idea reused | Same CSR layout (`frag_offsets`, `locus`, `state`) in `data/fragments.py`, but on CPU/mmap, per sample |
| `data/cpg_index.py` (wgbstools index from FASTA) | moved to `legacy/` | Only needed to read wgbstools PAT files / rebuild the genome-wide functional artifact |
| `scripts/build_genomewide_embedding.py`, `build_ext_locus_features.py` | moved to `legacy/representation_build/` | They **produced** the functional artifact we consume (provenance); they belong upstream in CpGRepresentationBenchmark, not in the downstream pipeline |

## 2. Code tied to the old framing (removed from the main pipeline)

| File | Old role | Problem under the new objective |
|---|---|---|
| `scripts/extract_reads.py`, `data/reads.py` | Loyfer PAT -> thinned reads | Tissue-reference data, not cfDNA |
| `scripts/build_sample_table.py`, `make_splits.py` | GSE186458 labels, proxy donor groups (lab, age, sex, race) | Dataset-specific; donor id was a proxy |
| `training/run.py` | per-read CE with sample label on every read; `cfdna_composition` (argmax -> bincount) | Per-read pseudo-labels; deconvolution by argmax counting is biased and not the endpoint |
| `training/tables.py` | 15 GB fp16 genome-wide table on GPU | GPU memory; replaced by materialising only observed loci, gathered per batch on CPU |
| `scripts/run_all_arms.sh`, `run_when_downloaded.sh` | orchestration of the above | obsolete |
| `docs/HANDOFF.md` | handoff notes for the Loyfer run | obsolete |
| `data/meta/{GSE186458_family.soft.gz, samples.parquet, splits.*}` | Loyfer metadata and split | not used; kept in git history (tag) |

## 3. Data / artifacts still useful

| Artifact | Location on this machine | Size | Use |
|---|---|---|---|
| `functional_pca_genomewide_hg38.h5` | `/raid/DATASETS/cfdna-transfer/cfdna-tissue-origin/data/` | 15 GB | **the `functional` arm**. Canonical `/cpg_idx` (locus_key = chrom_code<<32 \| 1-based pos, GRCh38) + `/embedding` fp16 [29,401,360 x 256], chunks (65536,256), no compression. Covers every CpG of chr1-22,X,Y. Built unsupervised (SVD on 1M random loci, seed 17), no methylation, no patients (see h5 attrs) |
| `hg38_cpg_index.npz` | same | 113 MB | wgbstools CpG order; only for legacy PAT parsing. Not needed by the new pipeline (lookups use locus keys) |
| CpGRepresentationBenchmark (code) | cloned to `/raid/DATASETS/cfdna-origin-work/CpGRepresentationBenchmark` | 22 MB | canonical h5 contract, NTv3-pre provider (for a future `ntv3_pre` artifact) |

## 4. Data / artifacts no longer needed by the primary pipeline

| Artifact | Size | Verdict |
|---|---|---|
| `data/reads/` (253 Loyfer npz, 1% thinned, >= 4 CpGs, wgbstools rows) | 9.4 GB | ARCHIVE — only for a possible secondary sanity check (reference-tissue separability); not cfDNA. See `DATA_CLEANUP.md` |
| raw Loyfer PAT (93 GB), `data/ext_store` (4.7 GB), locus feature store | not on this machine | not needed unless the functional artifact must be rebuilt (e.g. `functional_no_target_tissue`) |

## 5. Components reusable from CpGRepresentationBenchmark

- **Canonical HDF5 contract** `/cpg_idx` int64 + `/embedding` [N, D] with alias auto-detection
  (`representations/hdf5_store.py`); our `HDF5LocusEmbeddingStore` reads the same contract and the same locus-key
  namespace (`locus_key_chromcode<<32|pos_v1`, chrX=23, chrY=24), so any artifact materialised there is consumable
  here unchanged.
- **"Load only the needed subset" strategy** (their `HDF5RepresentationStore` gathers the needed rows once, sorted):
  replicated as per-dataset materialisation of observed loci.
- **NTv3-pre provider** (`providers/ntv3_pre.py`: InstaDeepAI/NTv3_650M_pre, 32,768 bp window, mean of the two
  central bins, 1536-d). Only a chr1 atlas exists there; a genome-wide / panel-restricted artifact must be
  materialised **in the benchmark repo** and consumed here via `configs/representations/ntv3_pre.yaml`.
- Functional PCA recipe (`build_functional_pca_embedding.py`) — identical recipe to the artifact above.

## 6. Methodological problems / leakage in the old pipeline

1. **Per-read pseudo-labels.** Every read of a sample received the sample label. For purified cell types this is
   approximately right; for plasma it is wrong (most cfDNA is haematopoietic). Must not be carried over.
2. **`argmax -> bincount` deconvolution** is biased toward over-confident classes and is not calibrated; it is not
   an estimator of tissue fractions.
3. **Model selection on per-read macro-F1** of val reads: selects for read-level separability, not sample-level.
4. **Proxy donor split** (lab, age, sex, race) could merge but not separate donors; 11 organs had one donor group
   (train-only), 2 had no test donor: the test set did not cover the label space.
5. **Sex chromosomes in the universe** (chrX/Y). With class-imbalanced sex (common in cancer cohorts), chrY
   read presence or chrX methylation can predict the label without any tissue signal. New default: autosomes only.
6. **15 GB table on GPU** forced a single-arm-per-GPU regime and prevented the `random` arm from being generated
   with the same memory profile.
7. **Coordinate memorisation** was untested: with a per-locus embedding table the classifier can learn
   "locus X -> class" irrespective of the representation. New controls: `position_only`, `random`,
   `methylation_only` and an optional unseen-locus evaluation (`docs/REFACTOR_PLAN.md`).
8. **External cfDNA** evaluated only qualitatively (top-5 composition), no labels: no endpoint.

## 7. Disk constraints on this machine (2026-09-30)

`/home` is 100% full (~270 MB free) and `/raid` is 99% full (~190 GB free). All data, caches and outputs go to
`/raid/DATASETS/cfdna-origin-work/` via `configs/paths.local.yaml` (gitignored); nothing large is written in the repo.
