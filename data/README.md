Default `data_root` (see `configs/paths.yaml`). Nothing here is versioned. Expected layout:

    <data_root>/<dataset>/meta/        public metadata (GEO SOFT, ENA run report, paper supplementary tables)
    <data_root>/<dataset>/raw/         per-sample FASTQ/BAM (large intermediates are deleted after extraction)
    <data_root>/<dataset>/processed/   samples.parquet, loci.npy, dataset_manifest.json, fragments/<sample_id>/
    <data_root>/reference/hg38/        GRCh38 FASTA + Bismark index
    <data_root>/cache/representations/<dataset>/<representation>/   materialised locus tables
