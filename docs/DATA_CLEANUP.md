# Data cleanup list (large, unversioned artifacts)

Nothing listed here was deleted automatically. Raw or expensive data is never deleted by this refactor. The only
in-repo deletions were git-tracked files that can be regenerated or are obsolete (recoverable from tag
`legacy-loyfer-v0`). Status as of 2026-09-30 on the current server.

| Path | Size | Content | Verdict | Reason |
|---|---|---|---|---|
| `/raid/DATASETS/cfdna-transfer/cfdna-tissue-origin/data/functional_pca_genomewide_hg38.h5` | 15 GB | `functional` representation, all 29.4M hg38 CpGs (chr1-22,X,Y), fp16 x 256 | **KEEP** | primary arm; about 1 GPU-hour to rebuild but needs the 18 GB feature store, which is not on this machine. It is a hard link to the original on the old server: do not modify in place |
| `…/cfdna-tissue-origin/data/hg38_cpg_index.npz` | 113 MB | wgbstools CpG order | ARCHIVE | used only by `legacy/` (PAT files, artifact rebuild) |
| `…/cfdna-tissue-origin/data/reads/` | 9.4 GB | 253 Loyfer-atlas samples, 1% thinned reads (old framing) | ARCHIVE | not cfDNA; possible secondary sanity check only. Delete if space is needed: it can be regenerated from the public GSE186458 PAT files with tag `legacy-loyfer-v0` |
| `…/cfdna-tissue-origin/data/meta/`, `.git`, code | <1 MB | old repo copy | DELETE (optional) | duplicated by git history (tag `legacy-loyfer-v0`) |
| `…/cfdna-transfer/MANIFEST.sha256`, `LEGGIMI_TRASFERIMENTO.txt` | small | transfer checksums | KEEP | lets you verify the 15 GB h5 (`sha256sum -c`) |
| `<data_root>/CpGRepresentationBenchmark` | 22 MB | shallow clone of the benchmark repo | KEEP | canonical HDF5 contract and NTv3-pre provider |
| `<data_root>/gse149438/meta/` | <1 MB | GEO SOFT, ENA run report, `samples_meta.tsv` | KEEP | regenerable with `prepare_gse149438.py metadata` |
| `<data_root>/gse149438/raw/<sample>/` | transient | FASTQ, trimmed FASTQ, Bismark BAM | DELETE (automatic) | removed per sample once its fragments are written (`keep_dedup_bam: false`) |
| `<data_root>/gse149438/processed/` | about 100-140 GB for 300 samples (pilot: 73 MB for 10.4M CpG calls, about 7 B/call after dropping `locus_key`; the median sample is about 6.5x the pilot) | fragment store | KEEP | costly to regenerate (realignment). It does not fit comfortably in the ~190 GB free on /raid alongside everything else: see the README note on disk |
| `<data_root>/reference/hg38/` | 15 GB | GRCh38 analysis set + Bismark index (+ `cpg_keys_autosomes.npy`) | KEEP while preprocessing | regenerable (`prepare_gse149438.py reference`, 33 min). `Bisulfite_Genome/*/genome_mfa.*.fa` (6 GB) are only used to build the index: DELETE. ARCHIVE/DELETE the rest when both datasets are processed |
| `<data_root>/envs/bioinfo`, `conda_pkgs/` | about 3 GB | bioinformatics conda env and package cache | KEEP env / DELETE `conda_pkgs` after install | the cache is regenerable |
| `<data_root>/scratch/` | <100 MB | research notes, ENCODE query tables | KEEP (small) | inputs of `docs/DATASETS.md` and `docs/ENCODE_LEAKAGE_AUDIT.md` |
| `<data_root>/cache/representations/` | about 190 MB per arm on 1.46M loci (PCA-64 fp16) | materialised tables | DELETE freely | regenerable in 10-40 s per arm |
| Old server: raw Loyfer PAT (93 GB), `data/ext_store` (4.7 GB), `locus_features_v1` store | — | inputs to rebuild the functional artifact | KEEP on the old server | needed for `functional_no_target_tissue` (see `ENCODE_LEAKAGE_AUDIT.md`) |
| `<data_root>/gse149438/geo_suppl/GSE149438_RAW.tar` | 6.1 GB | GEO processed methratio tables (hg19), raw public data | **KEEP** | input of the processed-beta pilot; read in place, never extracted |
| `<data_root>/gse149438_processed_beta/` | 2.5 GB | beta/coverage matrices 300 x 2,049,700 (GRCh38), manifests | KEEP | regenerable in about 5 min (`prepare_gse149438_beta.py`) |
| `<data_root>/cache/representations/gse149438_processed_beta/` | 262 MB per arm (PCA-64 fp16) | materialised tables | DELETE freely | regenerable |
| `<data_root>/outputs/gse149438_processed_beta/` | about 1-2 GB (estimate; mostly predictions and checkpoints) | pilot runs | KEEP | results of this phase |
| `<data_root>/pylibs/` | 228 MB | xgboost (no deps) | KEEP | optional dependency for `dmr_xgboost` |
