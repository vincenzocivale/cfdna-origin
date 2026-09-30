# CLAUDE.md

## Purpose
Train and evaluate a read-level tissue/cell-type-of-origin classifier for cfDNA WGBS reads. Each read is a set of
CpGs: (frozen locus embedding, methylation state 0/1, relative position) -> set encoder -> class logits.
Spin-off of `../CpGRepresentationBenchmark` (thesis: functional-annotation locus representations beat those
used in the literature). This repo consumes that repo's artifacts; it does not modify them.

## Invariants
- Locus embedding is `native_frozen`: never fine-tuned; only the classifier is trained.
- Split by DONOR, never by read (reads from one donor/sample must not span train and test).
- Fail explicitly if a read's CpGs are outside the embedding's locus universe; never silently narrow it.
- Same classifier, split, and budget for every embedding arm; only the embedding changes.
- Real cfDNA has no per-read labels: evaluate on in-silico mixtures and sample-level fractions, not per-read accuracy.

## Layout
- `src/cfdna_too/data/`: pat parsing, CpG index -> locus mapping, read table, donor splits, mixtures
- `src/cfdna_too/models/`: set encoder + classifier
- `src/cfdna_too/training/`, `evaluation/`
- `scripts/`: entry points; `configs/`: paths and experiment YAMLs
