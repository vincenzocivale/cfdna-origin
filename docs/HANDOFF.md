# Handoff: continuing training on another machine

State at handoff (2026-09-30): data pipeline complete, **no training results yet**. Only a 300-step smoke test ran (numbers
meaningless).

## What is in git
Code, configs, `data/meta/{samples,splits}.parquet` + `splits.json` (donor-group split, seed 20260929, assignment hash in
the json). Do not regenerate the split; use the committed one.

## What must be copied (not in git)
| Path | Size | Needed for |
|---|---|---|
| `data/reads/` (253 `.npz` + `.json`) | 9.4 GB | training/eval |
| `data/functional_pca_genomewide_hg38.h5` | 15 GB | `functional` arm |
| `data/hg38_cpg_index.npz` | 113 MB | everything (regenerable: `python -m cfdna_too.data.cpg_index --fasta hg38.fa.gz --out ...`) |

Not needed for training: raw PAT files (93 GB), `data/ext_store/` (4.7 GB), locus feature store. They are only needed to
rebuild reads/embedding (`scripts/extract_reads.py`, `build_ext_locus_features.py`, `build_genomewide_embedding.py`).

## Run
```bash
pip install -e ".[dev]"
PYTHONPATH=src python scripts/train.py --arm functional --seed 17   # then --arm none, --arm random; seeds 17 42 97
# or: scripts/run_all_arms.sh (sequential, skips arms with an existing results.json)
```
Results go to `outputs/<arm>/seed_<seed>/results.json`. Model selection uses val only; test and external cfDNA are
evaluated once at the end.

## Known issues / decisions
- **GPU memory:** the `functional` and `random` arms keep a 15 GB fp16 embedding table plus ~5 GB of reads on the GPU
  (~22 GB total). On the source machine another process used 17 GB, so nothing was launched. Use a GPU with >= 24 GB free,
  or move the table to CPU and gather per batch (not implemented).
- **Donor split is a proxy:** GEO has no donor id; `donor_group` = (lab, age, sex, race). It can only merge donors.
- 11 organs have a single donor group (train-only); Adipocytes and Skeletal have no test donor. See `splits.json`.
- Universe: chr1-22,X,Y (29,401,360 CpGs). chrM is excluded (435 CpGs; unsupported by the feature builder).
- `extract_reads.py` is resumable but treats any non-empty existing `.npz` as done; validate files after an interrupted run.
- External cfDNA (23) and matched WBC (23) samples are never used for training or model selection.
