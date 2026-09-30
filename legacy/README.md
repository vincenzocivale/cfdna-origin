# legacy/

Not imported by the pipeline. Kept for provenance only.

`representation_build/` holds the scripts that produced the `functional` artifact consumed by this repository
(`functional_pca_genomewide_hg38.h5`: TruncatedSVD(256) over 4165 binary ENCODE tracks + 23 dense annotation
features for every CpG of hg38 chr1-22,X,Y). They require the CpGRepresentationBenchmark locus feature store and the
MehylPredictor feature builder, neither of which is needed to run the benchmark. They should eventually move upstream
into CpGRepresentationBenchmark (representations are external artifacts here, see `docs/REFACTOR_PLAN.md`).
They are also the starting point for the future `functional_no_target_tissue` arm (`docs/ENCODE_LEAKAGE_AUDIT.md`).

The previous Loyfer-atlas read-level pipeline (per-read labels, in-silico mixtures, argmax->bincount composition)
was removed; it is available at git tag `legacy-loyfer-v0`.
