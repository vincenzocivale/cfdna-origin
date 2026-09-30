# cfdna-tissue-origin

Read-level tissue-of-origin classification of cfDNA WGBS reads using frozen CpG-locus representations
(primarily reference-genome functional annotations, PCA-compacted).

## Data
- Training/test reference: Loyfer et al. WGBS atlas (GSE186458, hg38): 253 samples, 39 cell types, 137 donors.
- Labels: cell type (39) and aggregated tissue, parsed from `GSM…_<CellType>-<DonorID>` filenames.
- Real cfDNA WGBS: not yet selected; evaluated via in-silico mixtures until then.

## Pipeline (planned)
1. CpG index (wgbstools order) -> hg38 coordinates from FASTA -> locus_key in the feature store
2. Genome-wide functional PCA embedding (27.08M loci x 256, fp16)
3. Read table: label, cell type, tissue, donor, sample, CpG ids, states, length (>= N CpGs)
4. Donor-level train/val/test split + manifest
5. Set-encoder training, per-read and mixture-level evaluation, baselines (random/ID embedding, CpGPT)

See `CLAUDE.md` for invariants and `configs/paths.yaml` for artifact locations.
