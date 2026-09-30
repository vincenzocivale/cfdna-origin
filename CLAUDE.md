# CLAUDE.md

## Purpose
Downstream benchmark for CpGRepresentationBenchmark. Question: do frozen functional (ENCODE) CpG representations
beat sequence-derived ones (and methylation-only) for **real plasma cfDNA sample -> tissue/cancer-of-origin**?
Datasets: GSE149438 (public, targeted bisulfite) and HRA003209 (MONITOR, controlled, EM-seq WMS).
This repo consumes the benchmark repo's representation artifacts; it does not rebuild or modify them.

## Invariants
- Representations are frozen artifacts (canonical `/cpg_idx` + `/embedding` HDF5) behind `LocusEmbeddingStore`;
  never fine-tuned, never a model parameter. The only representation-dependent module is the input adapter, and a
  common label-free PCA (r=64) keeps trainable parameters identical across arms.
- Same samples, split, fragments, preprocessing, model, budget, seeds, early stopping for every arm; hyper-parameters
  are fixed a priori, never tuned per arm.
- Supervision is sample-level (MIL). Never copy a sample label onto its reads; never evaluate per read.
- Split by patient, deterministic from the split seed, written once (assignment hash) and reused. Model selection on
  validation only; test labels hidden (`LabelGuard`) until the best checkpoint is reloaded; test evaluated once.
- Fail explicitly on loci outside a representation's universe, on missing data/artifacts; never fabricate samples.
- Autosomes only by default (sex-chromosome confounding). Record batch/hospital; report per batch.
- Synthetic data only in unit tests. Nothing large under the repo; paths via `configs/paths.local.yaml`.

## Layout
`src/cfdna_origin/{data,representations,models/{fragment,sample},training,evaluation,experiments}`,
`scripts/` entry points, `configs/{datasets,representations,models,experiments}`, `docs/`, `legacy/` (provenance).
Tests: `python -m pytest -q` (CPU, includes an end-to-end smoke test).
