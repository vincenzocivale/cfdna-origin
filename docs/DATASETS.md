# Real plasma cfDNA methylation datasets: research notes for the data adapters

Researched 2026-09-30. Every number below comes from a file in this directory tree or from a cited URL.
Anything I could not verify is marked **UNVERIFIED**.

Layout under `<data_root>/`:
- `gse149438/meta/`: `GSE149438_family.soft.gz`, `GSE149438_series_matrix.txt.gz`, `filelist.txt` (GEO suppl list),
  `ena_runs.tsv` (ENA filereport), `sra_runinfo.csv` (NCBI runinfo), `samples_raw.tsv` (parsed, 1 row per GSM),
  `PMC8595812_page.html` (paper HTML)
- `gse149438/suppl_example/GSM4502064_KRp1-10_bsmap_meth.txt.gz`: one processed file (22 MB), kept to show the format
- `hra003209/meta/`: `HRA003209_public_metadata.xlsx` (GSA public metadata), `indinstudy.json`/`runinstudy.json`
  (GSA ajax dumps), `individuals.tsv`, `runs_samples.tsv`, **`samples_split.tsv`** (1 row per patient: GSA accessions,
  BAM/FASTQ runs, diagnosis, stage, age bin, sex, **Training/Test group**, **hospital**, paper prediction scores),
  `PMC10533817.xml/.txt` (paper full text), `paper_supp/MOESM5_ESM.xlsx` (Supplementary Data 1-6), `paper_supp/MOESM6_ESM.zip` (Source Data)
- `scratch/parse_gse149438.py`, `scratch/parse_hra.py`, `scratch/parse_hra_supp.py`: the parsers that built the TSVs

---------------------------------------------------------------------------------------------------------------------

## Dataset A: GEO GSE149438 / BioProject PRJNA628686 / SRA SRP258729

### A.0 Identity: the premise was wrong
GSE149438 is **not** Stackpole et al. 2022 (cfMethyl-Seq). It is:
> Kandimalla R, Xu J, Link A, Matsuyama T, Yamamura K, Parker MI, Uetake H, Balaguer F, Borazanci E, Tsai S, Evans D,
> Meltzer SJ, Baba H, Brand R, Von Hoff D, Li W, Goel A. **"EpiPanGI Dx: A Cell-free DNA Methylation Fingerprint for the
> Early Detection of Gastrointestinal Cancers."** *Clin Cancer Res* 2021;27(22):6135-6144. PMID 34465601, PMC8595812,
> doi:10.1158/1078-0432.CCR-21-1982.

Sources: GEO SOFT `!Series_pubmed_id = 34465601` and PubMed esummary. Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8595812/.
(Stackpole's cfMethyl-Seq data is a different accession; I did not look it up.)

### A.1 Assay: targeted bisulfite capture, not WGBS and not MspI
Source: GSM `!Sample_extract_protocol_ch1` and paper Methods.
- 10 ng plasma cfDNA, bisulfite-converted with the Zymo Gold kit.
- **Swift Biosciences Accel-NGS Methyl-Seq** library (adaptase-based), 13 PCR cycles.
- Hybrid capture on **Roche NimbleGen SeqCap Epi Choice 30 Mb**, a custom panel called "gitBS": 67,832 tissue DMRs spanning
  25.6 Mb, taken from TCGA and GSE72872 450K data. 10 libraries were pooled per capture.
- NovaSeq 6000 (S4), stated as "paired-end 100 bp". The FASTQ does not match that: `SRR11615791_1` has about 50% 151-nt
  and 50% 100-nt reads, and runinfo `avgLength` is about 250-258 (both mates). Read length is therefore **mixed 100/151**,
  so trimming has to cope with both.
- No UMIs.
- Consequence: the CpG universe is the ~25.6 Mb capture target. Our locus embedding universe has to be intersected with the
  target BED. Per the CLAUDE.md invariant, off-target reads must be dropped explicitly and never silently. The target BED is
  paper Table S4/S3 ("gitBS panel design"). **I could not download it**: the PMC supplement links sit behind a JS
  proof-of-work page, and Europe PMC reports the paper is not open access. Get it by hand from
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8595812/ (NIHMS1738217-supplement-*.xlsx), or rebuild it from on-target coverage.

### A.2 Processed files on GEO: per-CpG only, no read-level data
- Series suppl: `GSE149438_RAW.tar` (6,143,518,720 B, about 5.7 GB). It contains 300 files, `GSM*_KRp{1,2}-N_bsmap_meth.txt.gz`,
  of 14-23 MB each (listed in `filelist.txt`). https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149438/suppl/
- Format, verified on GSM4502064: BSMAP `methratio.py` output with columns
  `chr pos strand context ratio eff_CT_count C_count CT_count rev_G_count rev_GA_count CI_lower CI_upper`.
  Every row is `CG` with strand `+`, and rev_* is NA, so strands are collapsed. The file has 1,883,516 CpGs on **hg19**.
  Per GEO, CpGs covered by fewer than 4 reads were removed.
- **These are per-CpG beta and count tables. They carry no per-fragment patterns.** Read-level data means realigning the SRA FASTQ.
- Original pipeline (GEO data_processing): adapter and quality trimming, then **BSMAP 2.90 -> hg19**, then methratio.py.
  No dedup step is mentioned (**UNVERIFIED** whether one was done).

### A.3 SRA / ENA
`ena_runs.tsv` and `sra_runinfo.csv` each have **300 runs**: exactly 1 SRR per GSM, all PAIRED, Bisulfite-Seq, NovaSeq 6000.
Run range SRR11615790..; experiments SRX8181972..
- Read pairs per sample: median 13.75 M, mean 14.4 M, range 2.13-45.8 M. Total bases 1,103.9 Gb.
- **ENA FASTQ total: 224.1 GB gzipped**, served directly (`fastq_ftp` column). You do not need sra-tools: aria2c or wget on
  `ftp.sra.ebi.ac.uk/vol1/fastq/SRR116/...` works.
- FASTQ GB by class: PDAC 53.1, ESCC 36.0, CRC 32.3, GC 32.2, Normal 31.3, HCC 30.9, EAC 8.3.

### A.4 Samples (`samples_raw.tsv`, 300 rows, all plasma cfDNA)
Columns: gsm, title, source_name, tissue (diagnosis), age, stage, alternate_id, assayed molecule, biosample, srx, srr, read_pairs,
bases, fastq_bytes, supp_file, title_prefix.

| class | n | KRp1 | KRp2 | stage available |
|---|---|---|---|---|
| PDAC | 74 | 24 | 50 | 44 (IIA 9, IIB 21, III 5, IV 9); 30 missing |
| ESCC | 48 | 30 | 18 | none |
| Normal (healthy) | 46 | 37 | 9 | n/a |
| HCC | 43 | 28 | 15 | 36 (I 20, II 9, IIIA 6, IIIB 1); 7 NA |
| CRC | 40 | **0** | **40** | 40 (0:2, I 4, II 10, III 13, IV 11) |
| GC | 37 | 17 | 20 | none |
| EAC | 12 | 8 | 4 | none |
| total | 300 | 144 | 156 | |

- Sample type: all 300 are `assayed molecule: cell-free DNA`, plasma. **No tumour tissue and no WBC samples were sequenced.**
  The tissue component of the paper is public TCGA 450K data.
- One sample per patient. `alternate id` (for example `HCC_43`) is unique across the 300 and works as the donor ID.
- Not recorded in GEO: **sex, batch/site, train/test assignment.**
- Age means by class: Normal 57.7, EAC 54.6, ESCC 63.1, GC 64.9, HCC 66.3, PDAC 66.9, CRC 68.9.
- `title_prefix` KRp1/KRp2 looks like a submission, library or sequencing batch. Its meaning is **UNVERIFIED**, but it is the
  only batch proxy available.

### A.5 Paper protocol
Source: PMC8595812 Methods, read via WebFetch.
- Cohort: "CRC (40), PDAC (74), HCC (43), EAC (12), ESCC (48), GC (37), normal (46)", "collected from various institutes".
  Only PDAC has its sites named: University of Pittsburgh 58, Medical College of Wisconsin 16. Sites for the other classes are
  **UNVERIFIED**. Author affiliations (Japan: Tokyo/Kumamoto; Germany: Magdeburg; Spain: Barcelona; South Africa; Johns
  Hopkins; TGen) suggest the cohorts come from several countries.
- **ESCC and EAC were merged into one "esophageal" class** in the multi-cancer model ("given their high similarity"). The
  per-cancer panels report them separately (AUC 0.94 ESCC, 0.90 EAC).
- Evaluation: per cancer, cancer plus normal samples were randomly split 70/30 into train and test, with no stratification
  mentioned. DMR calling (metilene) and Boruta feature selection ran on the training set only, with 10-fold CV for tuning.
  The whole split was repeated **10 times**. Multi-cancer: each class was split 70/30 independently, with one-vs-rest random
  forests. **The per-sample split is not published**, so it cannot be reproduced. Use our own donor-level splits.
- Features: mean methylation ratio of the CpGs in each DMR. Region-level, not read-level.

### A.6 Recommended route from public accessions to per-read patterns on hg38
Tools to install, none of which are present now: `trim_galore 0.6.10` (cutadapt 4.x), `bwa-meth 0.2.7` with `bwa-mem2`, or
`Bismark 0.24.x` with `bowtie2`; `samtools 1.20`; `picard MarkDuplicates` or `samtools markdup`; `MethylDackel 0.6.x`, or
`wgbstools` (bam2pat, which needs `wgbstools init_genome hg38`). sra-tools is not needed because ENA has FASTQ.
1. `aria2c` the 600 FASTQ.gz files (224 GB) into `/raid/.../gse149438/fastq/`.
2. Trim with `trim_galore --paired --clip_r2 10 --three_prime_clip_r1 10` (use `--clip_r1 10 --three_prime_clip_r2 10`
   as well if in doubt). This is Swift Accel-Methyl adaptase-tail removal, and the read-length mix is handled by
   quality/adapter trimming. (Standard Swift guidance. The exact trimming used in the paper is **UNVERIFIED**.)
3. Align to hg38 with `bwa-meth` (or Bismark), coordinate-sort, then **deduplicate**. Without UMIs, 10 ng input, 13 cycles
   and capture will give high duplication.
4. Keep MAPQ≥20 and proper pairs, restrict to the capture BED, and optionally clip mate overlap (`bam clipOverlap`, or
   `wgbstools` handles pairs natively).
5. Get per-fragment CpG patterns with `wgbstools bam2pat` (CpG-index `.pat.gz`, matching the pat parser planned in
   `src/cfdna_too/data/`), or with our own parser over XM/YD tags.
6. Map CpG index to locus universe, with an explicit failure or report for reads outside it.
- Compute and disk (estimates, **UNVERIFIED**): about 14 M pairs per sample on bwa-meth at 16 threads takes roughly
  0.5-1 h per sample, so 300 samples is about 150-300 wall-h at 16 threads (about 2.5-5k core-h). Sorted BAMs of about
  1-2 GB per sample come to roughly 0.4-0.6 TB, and pat files are much smaller. Peak disk with FASTQ + trimmed + BAM is
  about 1 TB on /raid. Delete trimmed FASTQ after alignment.
- A shortcut with no read-level data: the GEO methratio files (hg19 per-CpG) could support sample-level beta baselines
  after hg19->hg38 CpG liftover. They cannot feed the read-level set encoder.

### A.7 Confounder risks for Dataset A
- **Class is confounded with batch:** CRC is 40/40 in KRp2, Normal is 37/46 in KRp1, and PDAC is 50/74 in KRp2. A classifier
  can learn batch. Report per-batch results and consider batch-stratified evaluation.
- **Class is probably confounded with site and country** (PDAC is from US centres; the other sites are unknown). Healthy
  controls come from an unknown source.
- Age: controls are about 9-11 years younger than most cancer groups.
- EAC n=12 is too small to stand alone. Merge it with ESCC as the paper did, or drop it.
- Targeted panel: the CpG universe is DMR-selected from cancer vs normal tissue, so it is biased toward the question.
  Embedding arms are compared on that restricted universe.
- Depth varies 2.1-45.8 M pairs. Downsample or cap reads per sample.
- Tissue-of-origin here means cancer type from plasma. There are no pure reference cell types. For read-level labels, in-silico
  mixtures would need reference methylomes from elsewhere.

---------------------------------------------------------------------------------------------------------------------

## Dataset B: GSA-Human HRA003209 / BioProject PRJCA012255 (controlled access)

### B.0 Identity: confirmed
> Bie F, Wang Z, Li Y, Guo W, Hong Y, Han T, Lv F, Yang S, Li S, Li X, Nie P, Xu S, Zang R, Zhang M, Song P, Feng F, Duan J,
> Bai G, Li Y, Huai Q, Zhou B, Huang YS, Chen W, Tan F, Gao S. **"Multimodal analysis of cell-free DNA whole-methylome
> sequencing for cancer detection and localization."** *Nat Commun* 2023;14:6042. doi:10.1038/s41467-023-41774-w,
> PMID 37758728, PMC10533817. Open access (CC-BY).

MONITOR = "Multi-Omics Noninvasive Inspection of TumOr Risk", a multicentre case-control study. The method is called THEMIS.
The data availability statement names HRA003209 ("The 1277 cfDNA WMS data of the MONITOR cohort ... HRA003209").
GSA page: https://ngdc.cncb.ac.cn/gsa-human/browse/HRA003209. Study title "a multicancer early detection platform integrating
multimodal information from cell-free DNA whole-methylome sequencing", released 2023-08-16, submitted by Gao Shugeng
(Cancer Hospital, CAMS). The GSA page showed 34 access requests.

### B.1 Assay
Source: paper Methods.
- **Enzymatic methyl-seq (NEB EM-seq, E7120)**, not bisulfite. Plasma is 4 ml; all extracted cfDNA is used, capped at 30 ng,
  plus 100 ng carrier RNA. 9 PCR cycles. Unmethylated lambda spike-in; median conversion 99.4%.
- NovaSeq 6000, **PE100**, low-pass whole-genome. Analyses were downsampled to **60 M properly paired reads (~2x)**.
  773 samples were sequenced deeper, which the depth titration from 120 M down to 3 M uses.
- Pipeline: bcl2fastq 2.20, Trimmomatic 0.36, **Bismark 0.19.0 -> hg19** plus Bismark dedup, samtools 1.3, BamUtil 1.0.14
  clipOverlap, MAPQ≥20.
- Methylation feature (MFR): fragments with ≥3 CpGs, 80-250 bp, and non-CpG conversion >95%. Per 1-Mb window, MFR is the
  fraction of fully methylated fragments. So the paper already works at fragment level.

### B.2 Deposited files (public metadata: `HRA003209_public_metadata.xlsx`, `runs_samples.tsv`)
- 1,277 individuals and samples; 3,831 runs, all Illumina NovaSeq 6000, strategy "OTHER".
- Each sample has **1 BAM** (`HRRxxxxxx.bam`, run title `*_cfDNA_WMS`, median 7.57 GB, total **9.68 TB**) and **1 paired
  FASTQ run** (`HRRxxxxxxx_f1.fastq.gz` + `_r2.fastq.gz`, `*_cfDNA_WMS_fastq`, median 7.8 GB per sample, max 55.6 GB,
  total **12.78 TB**).
- The BAM contents are **UNVERIFIED**: presumably Bismark hg19 dedup'd alignments (with XM tags), and possibly the
  60 M-downsampled set, given the tight size distribution (IQR 7.49-7.63 GB). Check the header and tags once access is granted.
- Sample descriptions: "Plasma from healthy control N" / "Plasma from <NSCLC|COREAD|PACA|STAD|LIHC|BRCA|ESCA> patient N".
  Patient IDs are `GCH####` for healthy and `GCP####` for cancer. They match the paper's supplementary patient IDs 1277/1277.

### B.3 Cohort and split (paper Supplementary Data 4 joined to GSA: `samples_split.tsv`)
| Diagnosis | n | Training | Test | Hospitals (1-6) | Stage I/II/III/IV/NA |
|---|---|---|---|---|---|
| HEALTHY | 497 | 352 | 145 | 2 (235), 3 (262) | n/a |
| NSCLC | 157 | 110 | 47 | 2,3,4,5 | 32/16/20/35/54 |
| COREAD (CRC) | 150 | 105 | 45 | 2,5,6 | 11/23/40/60/16 |
| PAAD | 119 | 83 | 36 | 3,6 | 30/36/21/24/8 |
| GAC (gastric) | 114 | 78 | 36 | 5,6 | 4/18/32/42/18 |
| HCC | 113 | 78 | 35 | 1,6 | 41/20/11/25/16 |
| BRCA | 66 | 46 | 20 | 1 only | 2/14/12/25/13 |
| ESCA | 61 | 42 | 19 | 3,4 | 16/9/13/15/8 |
| total | 1277 | 894 | 383 | | |

- The split is a random 7:3 assignment of participants. **The per-sample Training/Test labels are fully recoverable**
  (Supp Data 4, `Group` column) and reproduce the paper's counts exactly. Hospital source (anonymised 1-6), age bin
  (17-45 / 45-60 / 60-92), sex and stage are included too. GSA sex matches the paper for all 1277.
- The paper also ran 100 random 7:3 resplits, and 384 leave-one-hospital-out combinations for detection only.
- Tissue-of-origin in the paper: localisation used methylation and fragmentation at TCGA ATAC tissue-specific peaks, evaluated
  on the test cohort. Accuracy figures: see the paper, Fig. 5 (**not extracted here**).
- There was also a separate WMS-vs-WGS concordance cohort (Supp Data 2: 220 healthy + 270 cancer). It is a subset or overlap
  of GCH/GCP IDs, and the WGS data are **not** in HRA003209 (**UNVERIFIED**).
- Paired tumour tissue and WBC exist for 65 patients (769-gene panel). They are not in HRA003209.

### B.4 Access procedure
Controlled access. Log in to NGDC (CNCB SSO) and click "Request Data" on the HRA003209 page. The request goes to
**DAC HDAC001831** ("cicams-Thoracic Surgery-GSG"; contact Bie Fenglong, biefl2021@163.com). The paper says: non-commercial
entities only, access within about 1 week, valid **1 year**. It normally needs a signed institutional application with a PI.
Download is by Aspera (`ascp -P33001 ... aspera01@download.cncb.ac.cn:gsa-human/`) or HTTP/FTP after approval.
Also note that data leaving China under human genetic resources rules may involve extra steps (**UNVERIFIED** for this DAC).

### B.5 Recommended route
- Cheapest: take the **BAMs** (9.7 TB, or a subset). If they carry Bismark XM tags, extract per-fragment CpG patterns
  directly: `wgbstools bam2pat` with an hg19 genome, or our own parser that merges mates. Then **lift CpG coordinates
  hg19 -> hg38** (UCSC chain, keeping only CpGs that remain CpGs in hg38, and failing or logging the rest explicitly).
- Cleanest: realign FASTQ (12.8 TB; about 30 M+ pairs per sample, far more for the 773 deep samples) to hg38 with
  bwa-meth or Bismark. EM-seq needs no special aligner mode. Trim with trim_galore (EM-seq: `--clip_r2 5`-ish is optional).
  Then dedup, clip overlap, and run bam2pat. The cost is roughly 10x Dataset A: about 1277 × 1-2 h at 16 threads.
  Stream per sample and delete FASTQ after alignment. Never stage all of it at once.
- Coverage is about 2x genome-wide, so per-read CpG sets are sparse per locus but spread over the whole genome. That suits a
  read-level model.

### B.6 Confounder risks for Dataset B (important for tissue of origin)
- **Diagnosis is heavily confounded with hospital.** BRCA comes 100% from hospital 1, and hospital 1 contains only BRCA+HCC.
  Healthy controls come only from hospitals 2 and 3, and are the majority there. Hospital 4 has only ESCA+NSCLC. Hospital 5
  has only CRC, GAC and NSCLC. A TOO classifier can pick up site effects. The paper's hospital hold-out analysis covered only
  cancer vs healthy detection. Use hospital-stratified or hospital-held-out evaluation where the design allows it (NSCLC and
  CRC span 3-4 sites), and report it.
- Age: in the paper's Supp Data 1, controls average 53±9 (train) and 49±9 (test) against 60±12 for cases. The age bins in
  Supp Data 4 look odd, for example PAAD 58/119 in the "17-45" bin, and should be treated with care (**UNVERIFIED** coding).
- Sex: BRCA is all female, ESCA and HCC are mostly male. Exclude chrX/chrY or check for sex leakage.
- Depth: FASTQ size varies 2.7-55.6 GB, and 773 samples were sequenced deep. Downsample to a fixed number of fragments.
  The fraction of FASTQ >15 GB is 0% for BRCA and 7-18% for the other classes.
- Stage: HEALTHY has no stage. NSCLC has 54/157 stage NA.
- All samples are from Chinese hospitals in a single study, which limits generalisation to Western cohorts such as Dataset A.
- The genome build is hg19 in both datasets, so hg38 work requires realignment or liftover in both.
