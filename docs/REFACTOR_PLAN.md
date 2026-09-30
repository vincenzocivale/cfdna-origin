# Refactor plan — MILESTONE_1_REFACTOR

## Scientific question
Do functional (ENCODE-annotation) CpG representations improve **real plasma cfDNA** tissue/cancer-of-origin
classification over sequence-derived CpG representations and over methylation-only models?

The CpG (locus) representation is the only experimental variable. Everything else — samples, splits, fragments,
observed CpGs, preprocessing, downstream architecture, trainable-parameter budget (as far as possible), optimiser,
budget, seeds, early stopping, aggregation — is shared by every arm.

## Scope decisions
- **Primary task:** real cfDNA *sample* -> class. Supervision at sample level only; no per-read labels.
- **No synthetic mixtures / deconvolution** in the main pipeline. The Loyfer-atlas code is removed (git tag
  `legacy-loyfer-v0`); only the scripts that built the functional artifact are kept in `legacy/` for provenance.
- **Arms for the current phase** (owner's decision): our functional representation vs methylation-only, plus the
  two cheap memorisation controls (`position_only`, `random`) which need no external artifact. `ntv3_pre` and
  `sequence` are configured as external artifacts and fail explicitly until the benchmark repo materialises them.
  NTv3 POST is never used. GENA-LM / Evo2 / CpGPT / MethylGPT: add a YAML pointing to a canonical h5.
- **Datasets** (identities verified, see `DATASETS.md`):
  - GSE149438 is EpiPanGI Dx (Kandimalla et al., Clin Cancer Res 2021). It is not cfMethyl-Seq. It is a SeqCap Epi
    targeted bisulfite panel with 300 plasma samples, one per patient. GEO holds only per-CpG hg19 tables, so the
    public FASTQ are re-aligned to GRCh38 with Bismark, streamed per sample.
  - HRA003209 is MONITOR/THEMIS (Bie et al., Nat Commun 2023), EM-seq WMS on hg19 Bismark BAMs. It is controlled
    access, so the repo has an adapter, validation that fails listing missing files, and an hg19->GRCh38 chain liftover
    that keeps only loci that are CpGs in GRCh38. No fake samples are created. The official Training/Test split is
    recovered from Supplementary Data 4; the paper's model scores in that table are dropped.
  - Confounders: GSE149438 batch KRp1/KRp2 (CRC 40/40 KRp2) and HRA003209 hospital (breast only in hospital 1,
    healthy only in hospitals 2-3). Splits stratify by label and batch, and results are reported per batch/hospital.

## Target layout
```
configs/{paths.yaml, datasets/, representations/, models/, experiments/}
src/cfdna_origin/
  config.py                 YAML loading, ${ENV} expansion, paths.local.yaml override
  data/                     sample table schema, fragment store (CSR, mmap), bam->fragments, splits, datasets/
  representations/          LocusEmbeddingStore (hdf5 | random | position | none), materialisation, manifests
  models/fragment/          Level-1 fragment encoders (Deep Sets, tiny Transformer)
  models/sample/            Level-2 aggregators (mean, gated-attention MIL, PMA) with exact streaming evaluation
  training/                 bag sampler, trainer (val-only model selection, early stopping)
  evaluation/               metrics, paired patient-level bootstrap / permutation, result I/O
  experiments/              run orchestration, provenance, RUN_COMPLETE
scripts/{prepare_gse149438.py, prepare_hra003209.py, materialize_representation.py, train.py, evaluate.py,
         run_benchmark.py, summarize_results.py}
legacy/                     representation builders that produced the functional artifact (not imported)
tests/                      unit + CPU end-to-end smoke test on tiny synthetic fixtures (tests only)
```

## Data model
- Per dataset: `samples.parquet` (sample_id, patient_id, label, label_fine, sample_type, batch, official_split,
  fragments_path, n_fragments, n_cpgs, ...) and one `fragments/<sample_id>.npz` per sample:
  `frag_offsets` int64 [F+1], `locus_key` int64 [C] (GRCh38, `chrom_code<<32 | 1-based C position`, the benchmark
  namespace), `state` uint8 [C]. CpGs of a fragment are position-sorted.
- `loci.npy`: sorted union of observed locus keys (coordinates only, label-free) — the dataset CpG universe.
- Universe: autosomes by default (sex-chromosome confounding); configurable. CpGs outside the representation
  universe raise, never silently dropped. Reads on non-universe contigs are filtered in preprocessing and counted
  in the dataset manifest (identical for every arm).

## Representation handling (GPU/memory)
- `materialize_representation.py` gathers `/embedding` rows for `loci.npy` from the canonical h5 in sorted order
  -> `cache/representations/<dataset>/<rep>.f16.npy` + manifest (source path, sha256 of the h5 header+attrs+sample
  rows, dim, build, normalisation). A targeted panel is a few hundred MB instead of 15 GB; WGBS stays on CPU as a
  memmap and rows are gathered per batch. The genome-wide table is never loaded on GPU.
- Common projection: every locus representation is reduced to r=64 by a PCA fitted on a seeded genome-wide sample of
  the *store itself*, not on the dataset or patients. It keeps the artifact's native variance structure, followed by
  a single global rescaling. On `functional`, PCA-64 retains 89% of the variance. No statistic is fitted on train or
  test data. Trainable parameters are identical across all embedding arms (80,716 in the primary model, of which
  4,288 in the adapter), and `methylation_only` has 76,492. The full Linear(D,d) adapter is a sensitivity analysis
  (`representation_projection: {type: none}`).
- `random`: deterministic hash-based Gaussian per locus key (same locus -> same vector in every dataset).
  `position_only`: chromosome one-hot + multi-scale sinusoidal coordinate code. `methylation_only`: no locus
  vector (a learned constant token).

## Model (details and literature in `ARCHITECTURE_REVIEW.md`)
Level 1: token = adapter(locus embedding) + state embedding + geometry(log gap, log offset, rank) -> small
permutation-invariant fragment encoder -> fragment embedding. Level 2: MIL aggregator over a bag of fragments ->
sample logits. Training: random bags of K fragments per sample per step; evaluation: deterministic, streamed over
all (or a fixed seeded subset of) fragments with an exact chunked softmax-attention accumulator.
Chosen (ARCHITECTURE_REVIEW §4):
- **Fragment encoder:** one SAB block followed by PMA-1, d=64.
- **Aggregator:** class-branch gated-attention MIL (Ilse 2018 / CLAM 2021).
- **Training:** bags of 8 samples x 4096 fragments, 4 bags per sample per epoch, AdamW 5e-4, weight decay 0.05,
  label smoothing 0.1, early stopping on validation macro-F1 with patience 15, at most 100 epochs.
- **Ablations:** mean pooling, PMA pooling, Deep Sets.

## Splits and leakage
Patient-level, stratified, deterministic from `split_seed` (independent of training seed); written once as a split
manifest with an assignment hash and reused. GSE149438: stratified K-fold outer CV (every patient predicted once
as test) with a stratified validation carve-out from the training folds. HRA003209: the paper's train/test split
(validation carved from train). Test labels are hidden from the training code path (`LabelGuard`) and test is
evaluated once, after best-checkpoint reload. Automated tests cover all items of the leakage checklist.

## Results
Each run: `config.yaml, dataset_manifest.json, representation_manifest.json, split_manifest.json, metrics.json,
predictions.parquet, training_history.json, checkpoint.pt, environment.json, RUN_COMPLETE.json` (last, only after
validation). `summarize_results.py` pools test predictions over folds, averages over seeds, and produces paired
patient-level bootstrap CIs for Δ macro-F1 / Δ balanced accuracy / Δ AUROC (+ paired permutation p-values,
reported descriptively).

## Unseen-locus generalisation (designed, not primary)
`evaluation.unseen_locus_fraction`: hash-partition the locus universe into train-visible / held-out loci
(deterministic on locus key); training bags drop CpGs on held-out loci, test evaluation is reported on all loci and on
held-out-only fragments. A representation that carries transferable information should degrade less than
`random` / `position_only`, which can only memorise.

## Milestone 1 exit criteria
Clean tree, new README, dataset adapters (GSE149438 prepare pipeline; HRA003209 schema+validation), representation
interface, architecture review, ENCODE leakage audit, config system, metrics/results system, green unit tests, tiny
CPU end-to-end smoke test. No GPU training before the architecture is confirmed.
