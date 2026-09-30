# CpG-set architecture selection (processed-beta pilot)

## Protocol (fixed before any run was made)
- **Candidates** (`configs/models/cpgset_{mean,deepsets,pma}.yaml`, `src/cfdna_origin/models/cpgset.py`). Each takes
  tokens `[adapter(PCA-64 locus embedding) | Linear(beta - 0.5, log1p(coverage)/5)]`, d = 32:
  - `mean`: GELU(Linear) token map -> mean pooling -> linear head
  - `deepsets`: 2-layer phi (hidden 64) -> mean pooling -> rho -> linear head
  - `pma`: 2-layer phi -> PMA (1 learned query, 4 heads) -> linear head
- **Development arms:** `methylation_only` and `position_only`, both representation-neutral. `functional`,
  `functional_shuffled`, `random` and `ntv3_pre` are never run during selection.
- **Data:** protocol `all_classes_original` (esophageal_merged), repetition seed 17 (split seed = 17), outer folds
  0-4. Selection uses **validation patients only**. The runs have `selection_only: true`, so test samples are never
  predicted and their labels are never revealed.
- **Training budget:** identical to the benchmark (`configs/experiments/_beta_common.yaml`).
- **Criterion:**
  - Primary: the mean over the 2 arms x 5 folds of validation macro-F1, with the best checkpoint evaluated on all
    observed CpGs of each validation sample.
  - Tie rule: if the best mean is within 0.01 of a simpler candidate, the simpler candidate wins
    (mean < deepsets < pma).
  - The winner is written to `configs/models/cpgset_selected.yaml` and then frozen for every arm, protocol and task.
- **Reproduce:**
  ```bash
  python scripts/run_beta_benchmark.py --experiments beta_archsel_mean beta_archsel_deepsets beta_archsel_pma --gpus 4,5 --procs-per-gpu 3
  python scripts/select_beta_architecture.py
  ```

## Result

Validation only (5 folds x 2 representation-neutral arms, seed 17):

| arch     |   val_macro_f1 |   val_macro_f1_sd |   val_bal_acc |   val_auroc |   params |   mean_best_epoch |
|:---------|---------------:|------------------:|--------------:|------------:|---------:|------------------:|
| mean     |         0.2172 |            0.0619 |        0.2652 |      0.5848 |     2470 |              24.5 |
| deepsets |         0.2362 |            0.0621 |        0.2933 |      0.5992 |     7750 |              25.6 |
| pma      |         0.2302 |            0.0498 |        0.2911 |      0.5956 |    14150 |              17.6 |

Validation macro-F1 per development arm:

| arch     |   methylation_only |   position_only |
|:---------|-------------------:|----------------:|
| mean     |             0.2061 |          0.2282 |
| deepsets |             0.2589 |          0.2136 |
| pma      |             0.2388 |          0.2215 |

**Selected: `deepsets`** (best mean 0.2362; tie margin 0.01; simplicity order mean < deepsets < pma). Frozen as `configs/models/cpgset_selected.yaml`.
