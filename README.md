# cfdna-origin

Downstream benchmark of the CpGRepresentationBenchmark project: **frozen CpG-locus representations** evaluated on
**real plasma cfDNA tissue/cancer-of-origin classification**.

## Scientific question
Do functional CpG representations (built from ENCODE annotations) improve real-cfDNA tissue-of-origin classification
over sequence-derived CpG representations and over methylation-only models?

## Primary datasets
| Dataset | Assay | Samples | Classes | Access | Split |
|---|---|---|---|---|---|
| **GSE149438** (EpiPanGI Dx, Kandimalla et al., *Clin Cancer Res* 2021) | targeted bisulfite capture (SeqCap Epi, 67,832 DMRs), PE | 300 plasma, 1 per patient | Healthy, Colorectal, Pancreatic, Hepatocellular, Gastric, Esophageal (ESCC+EAC merged by default; `esophageal_split` keeps them apart) | public (GEO/ENA) | patient-level stratified 5-fold outer CV (no published split) |
| **HRA003209** (MONITOR/THEMIS, Bie et al., *Nat Commun* 2023) | low-pass whole-methylome EM-seq, PE100 | 1277 plasma | Healthy, Lung, Colorectal, Pancreatic, Gastric, Hepatocellular, Breast, Esophageal | controlled (NGDC GSA-Human) | the paper's Training/Test split (Supplementary Data 4) |

Synthetic mixtures are not used for the benchmark. Known confounders are recorded and reported: batch KRp1/KRp2 in
GSE149438 (CRC is entirely KRp2) and hospital in HRA003209 (breast cancer comes only from hospital 1). See
`docs/DATASETS.md`.

## Experimental variable
Only the CpG (locus) representation changes. Every arm shares the same samples, splits, fragments, observed CpGs,
preprocessing, architecture, trainable-parameter count (via a common label-free PCA of each representation to r=64),
optimiser, budget, seeds (17, 42, 97), early stopping and aggregation. Representations are **frozen**. They are
external artifacts: canonical `/cpg_idx` + `/embedding` HDF5 files from CpGRepresentationBenchmark, accessed through
`LocusEmbeddingStore`.

| Arm | Content | Status |
|---|---|---|
| `functional` | TruncatedSVD(256) of 4165 ENCODE tracks + 23 annotation features, all hg38 CpGs | available |
| `methylation_only` | no locus vector (methylation state + fragment geometry only) | available |
| `position_only` | coordinate code (memorisation control) | available |
| `random` | frozen hashed Gaussian per locus (memorisation control) | available |
| `functional_shuffled` | functional vectors permuted across loci (content control) | available |
| `ntv3_pre` | NTv3-650M **pre** (never post) | needs materialisation in the benchmark repo |
| `sequence` | sequence-only baseline | needs materialisation in the benchmark repo |

To add a representation (GENA-LM, Evo2, CpGPT, MethylGPT, …), add a `configs/representations/<name>.yaml` that points
to a canonical HDF5.

## Primary endpoint
Sample-level classification: macro-F1, balanced accuracy, macro one-vs-rest AUROC and accuracy. Secondary metrics:
weighted F1, top-2 accuracy, per-class sensitivity/specificity/AUROC, confusion matrix and ECE. Arms are compared with
**paired patient-level bootstrap** 95% CIs of Δ (e.g. functional − methylation_only) and paired permutation tests.
Nothing is evaluated per read.

## Model
The model is hierarchical multiple-instance learning (MIL) with sample-level supervision only
(`docs/ARCHITECTURE_REVIEW.md`):
- **Tokens.** Each CpG token is the frozen locus embedding (PCA-64) plus the methylation state and the gap/offset/rank
  within the fragment.
- **Fragment encoder.** One set-attention block followed by PMA pooling produces a fragment embedding.
- **Sample aggregator.** A class-branch gated-attention MIL layer maps the fragment embeddings to sample logits.
- **Training.** Each step draws random bags of 4096 fragments per sample.
- **Evaluation.** Attention is computed exactly by streaming each sample in chunks.
- **Ablations.** Mean pooling, PMA pooling and a Deep Sets fragment encoder (`configs/models/`).

## Reproducibility: dataset → preprocessing → training → evaluation → summary
```bash
pip install -e ".[bam,dev]"                    # or: export PYTHONPATH=src
cp configs/paths.local.yaml.example configs/paths.local.yaml   # set data_root / outputs_root / representations_root

# 1. GSE149438 (public). Bioinformatics env (pin the Perl Bismark / trim_galore: the 3.x/2.x Rust rewrites differ):
#    conda create -p <env> -c conda-forge -c bioconda --override-channels bismark=0.25.1 trim-galore=0.6.11 \
#          bowtie2 samtools aria2 pigz        && export PATH=<env>/bin:$PATH
python scripts/prepare_gse149438.py metadata
python scripts/prepare_gse149438.py reference --threads 32
python scripts/prepare_gse149438.py process --workers 6 --threads 16   # streamed per sample; resumable; ~25-30 min/sample/worker
python scripts/prepare_gse149438.py finalize

# 1b. HRA003209 (controlled): metadata + an explicit list of what is missing
python scripts/prepare_hra003209.py metadata && python scripts/prepare_hra003209.py validate

# 2. Materialise the representations on the observed loci (optional; train.py does it lazily)
python scripts/materialize_representation.py --experiment gse149438_main

# 3. Train/evaluate: one run, or the whole grid (representations x seeds x folds; completed runs are skipped)
python scripts/train.py --experiment gse149438_main --representation functional --seed 17 --fold 0
python scripts/run_benchmark.py --experiment gse149438_main --device cuda:0

# 4. Summary: per-arm metrics and paired comparisons with CIs
python scripts/summarize_results.py --experiment gse149438_main
```
Each run writes the following to `{outputs_root}/<dataset>/<experiment>/<representation>/seed_<s>/fold_<k>/`:
`config.yaml`, `dataset_manifest.json`, `representation_manifest.json`, `split_manifest.json`,
`training_history.json`, `checkpoint.pt`, `predictions.parquet` (per-sample probabilities, read counts and CpG/read),
`metrics.json`, `environment.json` and finally `RUN_COMPLETE.json`. The last file is written only after the best
checkpoint has been reloaded, the test set evaluated once, and the predictions saved and validated.

## Layout
```
configs/{paths.yaml, datasets/, representations/, models/, experiments/}
src/cfdna_origin/{data, representations, models/{fragment,sample}, training, evaluation, experiments}
scripts/   entry points          tests/   unit + CPU end-to-end tests (synthetic fixtures only)
docs/      REFACTOR_PLAN, ARCHITECTURE_REVIEW, DATA_AUDIT, DATA_CLEANUP, DATASETS, ENCODE_LEAKAGE_AUDIT
legacy/    scripts that built the functional artifact (provenance only)
```

## Tests
```bash
python -m pytest -q
```
