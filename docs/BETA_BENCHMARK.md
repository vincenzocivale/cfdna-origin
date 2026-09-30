# Processed-beta pilot (`gse149438_processed_beta`): data mode and protocol

The goal is a cheap, controlled test of one question. Given the same methylation measurements, does the ENCODE
functional representation of a CpG carry tissue/cancer-of-origin information that genomic position, the embedding
distribution, the batch, or the methylation values alone do not explain? It uses the authors' public per-CpG
tables, so no reads are reconstructed. The result decides whether the costly fragment-level preprocessing is
worth running.

## Data
| Item | Value |
|---|---|
| Source | GEO `GSE149438_RAW.tar` (6.1 GB, 300 members `GSM*_KRp{1,2}-N_bsmap_meth.txt.gz`), read in place and never extracted or modified |
| Original format | BSMAP `methratio.py`: `chr pos strand context ratio eff_CT_count C_count CT_count rev_G_count rev_GA_count CI_lower CI_upper`. One row per CpG, strands collapsed onto the `+` C (`rev_*` = NA) |
| Genome build | hg19. Lifted to GRCh38 with the UCSC `hg19ToHg38.over.chain.gz` |
| Coordinate convention | 1-based position of the CpG C on the + strand, in both hg19 and GRCh38. The locus key is `chrom_code << 32 \| pos`, the CpGRepresentationBenchmark namespace. Verified empirically: only 250 of 2.05M lifted loci are not CpGs in GRCh38, whereas an off-by-one error would fail almost all of them |
| Contents as published | 547,364,007 rows in total. All are CG context, autosomes only, and CT_count >= 4 (the authors' threshold) |
| Filters applied | context CG; autosomes chr1-22; coverage (CT_count) >= 4. These equal the published filters, so 0 rows were removed |
| Values | beta = C_count / CT_count; coverage = CT_count (uint16; values above 65,535 would be clipped: 0 occurrences) |
| Missing CpGs | **Not imputed.** A CpG absent from a sample's table is NaN beta / 0 coverage in the dense matrices. Models see only observed (locus, beta, coverage) tokens. The classical baselines use region means over observed CpGs, with training-fold median imputation for logistic regression only (XGBoost handles missing values natively) |
| CpGs per sample | median 1,834,112 observed (IQR 1,796,401–1,871,065; min 968,802, max 1,952,250); missing fraction vs the union median 0.105 |
| Coverage per sample | median of per-sample mean coverage 74 (range 18–229) |

### hg19 -> GRCh38 liftover (`<processed_dir>/liftover_manifest.json`)
| n_input_loci (union of all samples) | n_lifted | n_failed (unmapped) | n_ambiguous | n_non_CpG_hg38 | many-to-one collisions | n_found_in_representation_store (functional) | n_final_loci |
|---|---|---|---|---|---|---|---|
| 2,052,381 | 2,051,650 | 731 | 0 | 250 | 1,700 | 2,049,700 | **2,049,700** (99.87%) |

Rules:
- Unmapped and ambiguous points (covered by more than one chain block) are dropped.
- On minus-strand blocks the target C is at the mapped base − 1.
- The target must be a CpG in the GRCh38 analysis set.
- Source loci that land on the same target are all dropped.
- The target must lie in every required representation store; `functional` covers 100% of the valid targets.

The union is lifted once, so every sample uses the same mapping.

## Task and protocols
- Sample-level supervision only: sample -> class. There are no per-CpG or per-read labels.
- Default scheme `esophageal_merged` (Healthy, Colorectal, Pancreatic, Hepatocellular, Gastric, Esophageal);
  `esophageal_split` is supported through `dataset_overrides: {label_scheme: esophageal_split}`.
- Protocols (see `GSE149438_BATCH_AUDIT.md`):
  - **A.** `all_classes_original`
  - **B.** `exclude_crc`
  - **C.** `batch_robust_subset`: classes with at least 8 patients in both batches, evaluated **cross-batch**.
- Negative control: batch prediction (KRp1 vs KRp2) with the same inputs (`beta_batchpred_*`).

## Splits
- Repeated, stratified, patient-level 5-fold CV. Repetition r uses split seed = training seed r ∈ {17, 42, 97};
  within a fold, validation is 20% of the non-test patients.
- Stratification is by class × batch. A stratum with fewer patients than folds falls back to class-only. CRC×KRp1
  does not exist and is reported as `classes_confounded_with_batch` in every split manifest; it is never forced.
- Splits are materialised once per (protocol, task, seed), saved as parquet plus a manifest with an assignment hash,
  and shared by every arm.

## Arms (identical downstream model and budget)
| Arm | Locus information |
|---|---|
| `methylation_only` | none (learned constant locus vector) |
| `summary_only` | no tokens: logistic regression on n observed CpGs, mean/variance beta, coverage mean/median, missing fraction |
| `position_only` | chromosome one-hot + multi-scale sinusoids of the coordinate, then PCA-64 |
| `random` | frozen hashed N(0,1) vector per locus, then PCA-64 |
| `functional_shuffled` | the functional PCA-64 rows permuted across the 2,049,700 loci. It is a bijection fixed by seed 1: the same matrix, marginal distribution and dimension, identical in train and test. Only the locus ↔ annotation link is destroyed |
| `functional` | ENCODE functional representation, then PCA-64 |
| `ntv3_pre` | configured; fails with an explicit message for this arm only until it is materialised |
| `dmr_logistic`, `dmr_xgboost` | embedding-free EpiPanGI-style baselines (below) |

PCA is fitted per representation on 200,000 seeded genome-wide loci of the store itself: no labels, no beta values.
Its dimension (32, 64 or 128) is one experiment-level setting for all arms. Each run records in
`representation_manifest.json`: `raw_dim`, `projected_dim`, `explained_variance_ratio`, `fit_universe_size`,
`pca_hash`, `source_representation_hash`.

## Compute-reduced grid (as run, single GPU)
| Experiment | Neural arms | Baselines | Repetition seeds x folds |
|---|---|---|---|
| A `beta_all_classes` | methylation_only, position_only, functional_shuffled, functional | summary_only, dmr_logistic, dmr_xgboost | 3 x 5 |
| B `beta_exclude_crc` | same | same | 1 x 5 |
| C `beta_batch_robust` | same | same | 3 x 2 (cross-batch) |
| batch control `beta_batchpred_exclude_crc` | methylation_only, functional_shuffled, functional | summary_only | 1 x 5 |

- `random` is configured but was not run. Locus identity is covered by `position_only`, and the embedding
  distribution by `functional_shuffled`.
- `beta_batchpred_all_classes_original` is configured but was not run. With CRC included, the class alone predicts
  KRp2, so `exclude_crc` isolates batch information better.
- `ntv3_pre` is skipped with an explicit message until it is materialised.

## Model
A CpG-set -> sample model (`models/cpgset.py`). The architecture was chosen once on validation data with
representation-neutral arms (`BETA_ARCHITECTURE_SELECTION.md`) and then frozen.

Training:
- Batches of 8 samples x 16,384 random observed CpGs; class-balanced sampling; AdamW 1e-3, weight decay 0.05;
  label smoothing 0.1.
- Early stopping on validation macro-F1 (patience 12, at most 60 epochs), with a fixed 262,144-CpG subset per
  validation sample.

Test samples are predicted on **all** of their observed CpGs, streamed exactly.

## EpiPanGI-style baselines and deviations
- **Paper protocol** (Kandimalla 2021): metilene DMR calling on the training set, Boruta feature selection with 10-fold
  CV, random forests, and repeated random 70/30 splits per cancer.
- **Here:**
  - Candidate regions are runs of consecutive panel CpGs with gaps <= 150 bp. This is label-free, approximates the
    panel's capture regions, and is fixed once for all folds.
  - Region beta = sum of methylated calls / sum of coverage.
  - Inside each fold, **on training+validation patients only**:
    - keep regions observed in >= 80% of them;
    - run a Welch t-test per class vs rest;
    - keep the top 100 regions per class with |Δβ| >= 0.05;
    - take the union and train logistic regression (standardised, C = 1, balanced weights) or XGBoost (300 trees,
      depth 3, learning rate 0.05, subsample 0.8).
  - Validation patients are used for fitting because these baselines have fixed hyper-parameters and no model
    selection. The neural arms use validation for early stopping.
- **Deviations:**
  - t-test in place of metilene, and no Boruta: both are unavailable and slow.
  - Logistic regression / XGBoost in place of random forest, as the spec requests.
  - Our CV in place of their unpublished splits.
- The number of DMRs, their CpG counts, the thresholds and the hyper-parameters are recorded per run
  (`representation_manifest.json`).

## Reproduce
```bash
export PYTHONPATH=src:<data_root>/pylibs          # pylibs: xgboost installed with --no-deps (see README)
python scripts/prepare_gse149438.py metadata
python scripts/prepare_gse149438_beta.py --workers 16
python scripts/audit_gse149438_batch.py
python scripts/run_beta_benchmark.py --experiments beta_archsel_mean beta_archsel_deepsets beta_archsel_pma --gpus 4,5
python scripts/select_beta_architecture.py
python scripts/run_beta_benchmark.py --experiments beta_all_classes beta_exclude_crc beta_batch_robust \
       beta_batchpred_exclude_crc --gpus 4 --procs-per-gpu 4
python scripts/report_beta_benchmark.py
```
