# ENCODE track audit of the `functional` representation (leakage and confounding)

Status: audit written 2026-09-30. Nothing was retrained or recomputed for it. Every claim is tagged
**[verified]** (read from code, contracts or docs in this workspace), **[reconstructed]** (derived from the public ENCODE
portal, not from the frozen track list) or **[unverified]** (an inference that still needs checking).

## 1. Bottom line

- The frozen 4,165-track list (`regulatory/track_contract.tsv`) **is not available on this machine**. It is not in this
  repo, not in `MehylPredictor/` (the builder), and not in the benchmark shallow clone. The store it lives in
  (`locus_features_v1`, symlinked from `MehylPredictor/local_methyl_data/derived/`) is on another host. Its upstream
  source (`derived/ntv3_functional_peak_atlas_chr1_all_sources/all_primary_columns.tsv`, originally under
  `/dune/DATASETS/MethylPredictionData/`) is not mounted either. **[verified]**
- The code that chose the tracks (the ENCODE query and biosample filters) is **not in any git history** we could search:
  159 MehylPredictor commits, `git log -S` for `encodeproject.org`, `all_primary_columns` and `functional_peak_atlas`.
  The builder treats the contract as frozen provenance: "URLs and MD5s are provenance, never a new file-selection
  procedure" (`MehylPredictor/docs/data/LOCUS_FEATURES.md`). **[verified]**
- So the per-organ counts in Section 4 are **reconstructed** from the ENCODE portal. They describe the *candidate
  universe* the tracks were drawn from, not the exact frozen selection. They are **not** exact counts of the frozen tracks.
- What is certain: the embedding encodes reference epigenome priors from ENCODE, which include target-tissue and
  target-cancer cell-line tracks (liver/HepG2, colon/HCT116, and so on). No cfDNA sample, patient or diagnosis is used.
  That means **no label leakage** in the strict sense. There is a **confound**: we cannot tell whether functional
  annotations help in general or whether annotations from the target tissues are what helps.

## 2. What enters the embedding [verified]

Recipe: `legacy/representation_build/build_genomewide_embedding.py`, previously `scripts/build_genomewide_embedding.py`.
It is the same recipe as the benchmark's `build_functional_pca_embedding.py`. Each CpG gets a 4,188-dimensional vector:

`[4165 binary ENCODE peak-overlap bits × global unit scale | 23 dense features z-scored]`.

`TruncatedSVD(256, seed 17)` is fit on 1,000,000 seeded random loci, spread across chromosomes in proportion to their
size. It is then projected onto all 29,401,360 hg38 CpGs (chr1-22, X, Y). The fit uses no methylation values, no
patients and no labels (`attrs: supervision="none", patient_specific=False`). The track block uses one scalar scale,
`1/sqrt(mean(bit^2))`, not per-track standardization. As a result, frequently-on tracks (broad marks, promiscuous
accessibility) carry more variance than sparse tracks.

### 2.1 Dense core features (18): `MehylPredictor/resources/locus_features/annotation_core_v1.json`

| idx | name | source |
|---|---|---|
| 0-3 | `cpg_context__{island,shore,shelf,open_sea}` (one-hot; shore = 2 kb, shelf = next 2 kb) | UCSC cpgIslandExt hg38 |
| 4-7 | `genomic_region__{promoter,exon,intron,intergenic}` (promoter = TSS −2 kb/+500 bp, strand-aware) | GENCODE v50 |
| 8-16 | `ccre_class__{none,PLS,pELS,dELS,CA,TF,CA-CTCF,CA-H3K4me3,CA-TF}` (one-hot) | ENCODE cCRE Registry V4 |
| 17 | `dist_tss_abs_log10__z`, z-scored with frozen mean 3.2289 and SD 0.9242 (fit on historical TCGA training CpGs) | GENCODE v50 |

The cCRE Registry V4 classes are **cell-type-agnostic** aggregates over all ENCODE biosamples. They carry no tissue
identity. The TSS z-score parameters come from TCGA array CpG positions (coordinates only, no methylation values), so
they contain no cfDNA information.

### 2.2 Breadth features (5): `MehylPredictor/resources/locus_features/breadth_v1.json`

Each is the fraction of tracks in a fixed group that have a peak overlapping the CpG:

| idx | name | formula |
|---|---|---|
| 0 | `accessibility_breadth` | overlapping accessibility tracks / 533 |
| 1 | `histone_breadth` | overlapping histone tracks / 1959 |
| 2 | `ctcf_breadth` | overlapping CTCF tracks / 201 |
| 3 | `tf_binding_breadth` | overlapping TF ChIP tracks / 1472 |
| 4 | `core_breadth` | overlapping "core" tracks / 2401 (accessibility + CTCF + histone marks H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9ac, H3K9me3) |

These are tissue-agnostic in form, since they count biosamples without naming them. They are still **weighted by the
biosample composition** of each group. If blood or HepG2 dominate a group, "breadth" partly means "active in blood or
HepG2".

### 2.3 ENCODE peak tracks (4,165): what is verified from code

From `MehylPredictor/src/methylation_predictor/locus_features/regulatory.py`, whose `load_track_contract` asserts all of
these:

- Exactly 4,165 rows, one ENCODE processed **peak** file (`ENCFF…bed.gz`) per track. There are no BigWig signals, no
  peak-score weighting, no window aggregation and no biosample aggregation. A bit is 1 if the CpG point `[pos-1, pos)`
  falls inside any peak.
- `functional_block` counts: **accessibility 533, histone 1,959, CTCF 201, tf_binding 1,472**.
- Every file has `assembly == GRCh38`, `status == released`, a unique accession, and a recorded MD5 and download URL.
  Sources are MD5-verified on download.
- Histone tracks carry a `histone_mark_group`. At least the seven core marks listed above are present. The remaining
  histone tracks use other marks (in ENCODE these are typically H3K4me2, H3K79me2, H4K20me1, H2AFZ, H3K9me2). **[unverified]**
- The contract has columns `track_index`, `file_accession`, `download_url`, `md5sum`, `file_size`, `functional_block`,
  `histone_mark_group`, `assembly` and `status`. Biosample columns may also exist, but the code does not read them. **[unverified]**
- The accessibility block is DNase-seq and/or ATAC-seq. The split between the two is **[unverified]**. The TF block is
  TF ChIP-seq minus CTCF, which has its own block.
- The name "all_primary" suggests one primary peak file per ENCODE experiment. "all_sources" suggests that all ENCODE
  award sources (ENCODE2/3/4, Roadmap, GGR) were eligible. "ntv3_" reflects that the atlas was first built as probe
  targets for NTv3 embeddings (`derived/ntv3_probe_targets/…`). All of this is **[unverified]**.

## 3. Origin of the tracks

- All tracks are public ENCODE portal files (hg38). They are **reference epigenomes** from ENCODE/Roadmap biosamples:
  cell lines, primary cells, tissues, in-vitro differentiated cells and organoids.
- The datasets we classify are GSE149438 cfDNA WGBS (Healthy plus colorectal, pancreatic, hepatocellular, gastric and
  esophageal ESCC/EAC cancers) and MONITOR/HRA003209. Neither is deposited in ENCODE, so neither can be a source of
  tracks. ENCODE tissue donors (for example the ENTEx four-donor series and Roadmap donors) are different individuals
  from the Chinese cfDNA cohorts. **[verified for ENCODE content types; the cohort identity is by construction]**

## 4. Tracks per organ and biosample (reconstructed)

**Method.** We queried the ENCODE REST API on 2026-09-30:
`type=Experiment&status=released&assembly=GRCh38&organism=Homo sapiens`, with `assay_title` ∈ {DNase-seq, ATAC-seq,
Histone ChIP-seq, Mint-ChIP-seq, TF ChIP-seq}. The raw output is in
`/raid/DATASETS/cfdna-origin-work/scratch/encode_audit/`, with scripts `tab.py` and `encode_universe.tsv`. Each
experiment was assigned to one organ group:

1. Known cancer and blood cell lines by name, for example HepG2→liver, HCT116/Caco-2/LoVo→colon, Panc1→pancreas,
   K562/GM12878→blood.
2. Otherwise by `biosample_ontology.organ_slims`, in the priority order of the table rows.

The query returns **11,157 experiments**, compared with 4,165 frozen tracks. The frozen list is therefore a subset
(about 37%) selected by an unknown filter. A second subset that excludes ENCODE4 awards ("legacy") has block totals
closer to the contract: 776/2,269/249/2,094 vs 533/1,959/201/1,472. It is shown only as a plausibility bracket.
**Neither table gives the exact frozen counts.**

### 4.1 Full released GRCh38 universe (upper bound)

| group | accessibility | histone | CTCF | TF | total | % |
|---|---:|---:|---:|---:|---:|---:|
| liver | 28 | 75 | 11 | 857 | 971 | 8.7 |
| colon/intestine | 123 | 303 | 41 | 46 | 513 | 4.6 |
| stomach | 25 | 74 | 13 | 15 | 127 | 1.1 |
| pancreas | 35 | 85 | 15 | 10 | 145 | 1.3 |
| esophagus | 7 | 58 | 12 | 9 | 86 | 0.8 |
| blood/immune | 1454 | 1301 | 57 | 1021 | 3833 | 34.4 |
| lung | 150 | 242 | 42 | 263 | 697 | 6.2 |
| breast | 142 | 48 | 13 | 169 | 372 | 3.3 |
| other | 1643 | 1873 | 253 | 644 | 4413 | 39.6 |
| ALL | 3607 | 4059 | 457 | 3034 | 11157 | 100 |

### 4.2 Legacy subset (ENCODE2/3, Roadmap, GGR, community; ENCODE4 excluded)

| group | accessibility | histone | CTCF | TF | total | % |
|---|---:|---:|---:|---:|---:|---:|
| liver | 10 | 57 | 7 | 321 | 395 | 7.3 |
| colon/intestine | 49 | 141 | 15 | 46 | 251 | 4.7 |
| stomach | 23 | 44 | 8 | 15 | 90 | 1.7 |
| pancreas | 10 | 49 | 6 | 10 | 75 | 1.4 |
| esophagus | 5 | 25 | 8 | 9 | 47 | 0.9 |
| blood/immune | 146 | 534 | 41 | 781 | 1502 | 27.9 |
| lung | 72 | 180 | 32 | 239 | 523 | 9.7 |
| breast | 16 | 35 | 12 | 161 | 224 | 4.2 |
| other | 445 | 1204 | 120 | 512 | 2281 | 42.3 |

If the frozen selection does not depend on organ (an **[unverified]** assumption), the expected frozen counts are about
the percentage times 4,165:

| group | expected tracks |
|---|---|
| liver | 300-360 |
| colon/intestine | about 190 |
| stomach | 45-70 |
| pancreas | 55-60 |
| esophagus | 35 |
| blood/immune | 1,150-1,430 |

### 4.3 Key biosamples (universe; legacy in brackets)

| biosample | origin | experiments |
|---|---|---|
| **HepG2** | hepatocellular/hepatoblastoma line; **liver cancer** | 845 [305]: TF 820 [284], histone 15, DNase 5, CTCF 5. The largest single contributor to the TF block. |
| **HCT116** | colorectal carcinoma | 202 [45] |
| Caco-2 | colorectal adenocarcinoma | 16 [5] |
| LoVo, HT-29, SW480 | colorectal | 1 each (DNase) |
| **Panc1** | pancreatic carcinoma | 16 [15] |
| 8988T, HPDE6-E6E7 | pancreas | 1 each |
| OE33, OE19, KYSE-* | esophageal cancer | **0**. No released GRCh38 experiments were found. |
| AGS, SNU-* | gastric cancer | **0** |
| **K562** | CML, blood | 916 [560] |
| **GM12878** | lymphoblastoid, blood | 209 [201] |
| primary T/B/NK, monocytes, spleen, thymus | blood/immune | about 2,700 more |
| A549 | lung adenocarcinoma | 360 |
| MCF-7 | breast | 235 |
| T47D | breast | 77 |

Tissue tracks for the target organs (ENTEx/Roadmap) include:

- liver: liver and right lobe of liver
- colon: sigmoid colon, transverse colon, colonic mucosa, rectal mucosa
- stomach: stomach, mucosa of stomach, gastroesophageal sphincter
- pancreas: pancreas, body of pancreas, endocrine pancreas
- esophagus: squamous epithelium, muscularis mucosa

These are mostly histone plus DNase/ATAC tracks. Gastroesophageal sphincter is assigned to stomach by `organ_slims`.

### 4.4 Overlap with the target classes

| target class | normal-tissue tracks | cancer cell-line tracks of the same origin | strength of prior |
|---|---|---|---|
| Healthy (blood-derived cfDNA) | blood/immune about 30% of the universe | K562, GM12878, HL-60 | **very strong** |
| Hepatocellular | liver tissue, hepatocytes | **HepG2** (dominant in TF) | **very strong** (TF-heavy) |
| Colorectal | colon/rectum tissues | HCT116, Caco-2, LoVo | strong |
| Pancreatic | pancreas tissues | Panc1 | moderate |
| Gastric | stomach tissues | none | weak |
| Esophageal (ESCC/EAC) | esophagus tissues | none (no OE33 or KYSE) | weak |
| MONITOR extras (lung, breast, etc.; the class list is **[unverified]**) | lung, breast | A549, PC-9, Calu3, MCF-7, T47D | strong for lung and breast |

The imbalance matters for interpretation. If `functional` helps more on HCC/CRC/Healthy than on gastric/esophageal, that
pattern is what target-tissue prior coverage would predict. It should not be read as evidence that annotations help in
general.

## 5. Interpretation

1. **What the features are.** They are a tissue-specific regulatory **prior** computed from reference epigenomes. They
   are not patient data. Using them is legitimate prior knowledge, and the thesis hypothesis is exactly that such priors
   help. Methylation-atlas deconvolution methods rely on the same kind of tissue-specific reference.
2. **Strict label leakage: none.** No cfDNA sample, diagnosis, split or methylation value enters the features or the SVD
   fit. The SVD is unsupervised and fit on random genomic loci. The only fitted dense parameter (TSS z-score) comes from
   TCGA probe coordinates. **[verified]**
3. **Indirect leakage through shared individuals: essentially nil.** ENCODE donors (ENTEx, Roadmap) and cell lines are
   not the GSE149438 or HRA003209 patients, and those cohorts are not in ENCODE. Beyond this, all that remains is shared
   biology (liver enhancers are liver enhancers). That is the intended prior, not leakage.
4. **The confound.** The track set covers the target organs and their cancer lines unevenly. A gain for `functional`
   over `none` or `random` therefore supports "reference functional annotations help". It does **not** separate
   "annotations help in general" from "annotations *of the target tissues* help". Only an arm that removes the target
   tissues (Section 6) can separate the two.
5. **Uneven compression by the SVD.** Tracks share a single global scale, so the leading components are driven by
   large, correlated biosample blocks: blood/immune, HepG2 TF ChIP, and ubiquitous promoter/CTCF signal. Small blocks
   such as esophagus (about 1%) and stomach (about 1-2%) may survive only in low-variance components or not at all in
   the 256 retained dimensions. Tissue-specific information is therefore represented unevenly, and the ordering is
   roughly the same as the class imbalance above. **[unverified; diagnostic in 6.4]**
6. **Breadth as a tissue-agnostic proxy.** The 5 breadth features count the fraction of biosamples in which a CpG falls
   in a peak (constitutive vs. restricted activity) without saying which tissue. They are the closest existing
   tissue-agnostic summary of the tracks, with the caveat about group composition in Section 2.2.

## 6. Proposed arms and diagnostics

### 6.1 `functional_no_target_tissue` (definitive test; needs the locus feature store)

Drop every track whose biosample matches a target class, then refit `TruncatedSVD(256)` with the same seed, the same
1M-locus sampling scheme and the same scaling, and project again. Keep the classifier, split and budget identical, as
required by CLAUDE.md. Tracks to drop, by `biosample_ontology` (organ_slims plus term names):

- **Hepatocellular:** organ_slim `liver` (liver, right/left lobe of liver, hepatocyte, hepatic stellate cell); HepG2,
  Hep G2, HuH-7, HuH-7.5, Hep3B, PLC/PRF/5.
- **Colorectal:** organ_slims `large intestine`/`intestine` (sigmoid/transverse/descending colon, colonic mucosa,
  rectal mucosa, large intestine). Also consider small intestine (duodenum, Peyer's patch), since organ_slims merge
  them. Cell lines HCT116, Caco-2, LoVo, HT-29, SW480, SW620, DLD-1, RKO.
- **Pancreatic:** organ_slim `pancreas` (pancreas, body of pancreas, endocrine pancreas, islet/β-cell and progenitor in
  vitro differentiated cells); Panc1, 8988T, Capan-1, HPDE6-E6E7.
- **Gastric:** organ_slim `stomach` (stomach, mucosa of stomach, stomach smooth muscle, gastroesophageal sphincter);
  AGS, SNU-* if present.
- **Esophageal:** organ_slim `esophagus` (squamous epithelium, muscularis mucosa, esophagus mucosa, epithelial cell of
  esophagus); OE33, OE19, KYSE-*, if present.
- **MONITOR extras [unverified class list]:** lung (A549, PC-9, Calu3, IMR-90 as lung fibroblast, lung tissues) and
  breast (MCF-7, T47D, MCF 10A, breast epithelium). Add further organs once the HRA003209 labels are confirmed.
- **Healthy/blood:** do **not** drop blood in the main arm. Plasma cfDNA is about 90% haematopoietic, and removing blood
  would test something different. Run `functional_no_blood` as a separate optional arm.

Implementation notes:

- Build the drop mask from the contract's biosample columns if they exist. Otherwise join `file_accession` against the
  ENCODE API (`/files/ENCFF…/?format=json` → `dataset` → `biosample_ontology`). The mask is fully determined by
  metadata, so write it to `configs/` together with the resulting per-organ counts.
- Recompute breadth on the reduced track set (new denominators). If the old breadth is kept, the dropped tissues leak
  back through the counts.
- Report the number of tracks dropped per block. Expect several hundred to about 1,500, driven mostly by HepG2 TF.
  Also run a **size-matched random-drop control** (drop the same number per block at random, 3 seeds). Without it, a
  loss cannot be told apart from simply having fewer tracks.
- Optionally, a leave-one-class-out variant drops only class *c*'s tissues and checks that class-*c* performance
  degrades the most.

**Cost.** Per locus the store holds 521 B packbits + 72 B annotation_core + 20 B breadth + 16 B key/position ≈ 630 B.
For 29.4M loci that is **about 18-19 GB** (27.1M main + 2.3M in `data/ext_store`, whose 4.7 GB includes its sources).
The 4,165 MD5-verified peak `.bed.gz` sources add **an estimated 5-15 GB [unverified]**; they are needed only if
breadth is recomputed from peaks, and breadth can instead be recomputed from the packbits. The store is **not on this
machine**, and `/home` is full. Compute is small: the SVD fit on 1M × about 3,000 sparse columns takes minutes, and a
streaming projection of 29.4M loci takes about 1 GPU-hour, reading the whole store once. The output is a new 15 GB fp16
H5 per arm.

### 6.2 Cheaper arms to run first

- **`functional_dense_only` / `functional_breadth_only`.** Use the 23 dense features, or only the 5 breadth features,
  z-scored. Feed them raw (23 or 5 dimensions, padded or linearly mapped to the model width as for other arms) or through
  a 23→k SVD. They need only `annotation_core.f32.npy` and `breadth.f32.npy` (about 2.7 GB in total, about 0.6 GB for
  breadth alone), not the packbits. This tests how much of the gain is tissue-agnostic.
- **`functional_tracks_shuffled`.** Note the design constraint: permuting track columns, or relabelling biosamples,
  leaves an SVD embedding unchanged up to rotation, so it is **not** a valid control. A valid null breaks the link
  between locus and track. Options:
  - Circularly shift each track's bit column along the genome by an independent random offset. This keeps peak density
    and width but destroys locus identity.
  - Permute rows of the track block within chromosome, keeping the dense block aligned.
  This still needs the packbits.
- **Masked projection without refit** (approximation). Zero the rows of `components` (saved in
  `*.projection.npz`) that belong to target-tissue tracks, then project again. This avoids the SVD refit but still needs
  the store. Use it only as a quick check of the refit arm.

### 6.3 Arms that need no store: none are exact

Every track-level ablation needs the per-locus bits. The existing 15 GB embedding cannot be edited to remove tracks.

### 6.4 Quick diagnostic, doable now given the contract and `*.projection.npz`

`functional_pca_genomewide_hg38.projection.npz` stores `components` [256, 4188] plus the scaling constants. It is about
4 MB and is not on this machine. With the track contract joined to ENCODE biosample metadata:

1. For each PC k, compute the share of squared loading mass per organ block: `sum_{t∈organ} W[k,t]^2 / sum_t W[k,t]^2`.
2. Plot the organ share against PC index and explained variance. Report the cumulative explained variance attributable
   to each organ block, using `explained_variance_ratio × share`.
3. Correlate per-PC loadings with organ-indicator vectors (point-biserial) to find PCs aligned with liver/HepG2,
   colon/HCT116, blood and so on, and check whether stomach and esophagus get any dedicated PC.
4. Later, relate this to the classifier through per-class gradient/attribution norms on the input embedding dimensions.

If stomach and esophagus carry near-zero loading mass, the arm in 6.1 is expected to barely change those classes.

## 7. Open items (unverified; to resolve when the store host is reachable)

- Get `locus_features_v1/regulatory/track_contract.tsv` (SHA-256 is in `regulatory/manifest.json`) and replace
  Section 4 with exact counts by joining `file_accession` to ENCODE biosample metadata.
- Confirm the accessibility split (DNase vs. ATAC), the non-core histone marks, and whether ENCODE4 or Roadmap files
  are included.
- Confirm the HRA003209/MONITOR cancer-type list and extend the drop map.
- Locate the original atlas-building script, which would contain the ENCODE query and filters. It may survive only in
  the backup of the deleted `derived/ntv3_post_track_manifest` (2.6 GB, listed in
  `MehylPredictor/configs/external_cleanup_policy.yaml`).
