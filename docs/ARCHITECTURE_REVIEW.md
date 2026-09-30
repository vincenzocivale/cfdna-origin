# Architecture review: sample-level cfDNA classification from bags of CpG-token fragments

Scope: choose the downstream model that is held fixed while the frozen CpG-locus representation is varied
(functional ENCODE-PCA 256-d, NTv3-pre, sequence, random, position-only, methylation-only). Task: real plasma cfDNA
sample -> cancer / tissue-of-origin class, sample-level labels only. Literature checked 2026-09-30.

Citation status: **[V]** = title/venue/year confirmed by web search during this review; **[K]** = well-established
reference cited from memory (standard, low risk) but not re-fetched; **[U]** = exists but some details (venue status,
architecture specifics) could not be confirmed and should be re-checked before citing in a manuscript.

## 0. Dataset facts (corrections to the working assumptions)

- **GSE149438 is not Stackpole et al.** It is **EpiPanGI Dx**: Kandimalla R, ..., Li W, Goel A. *EpiPanGI Dx: a
  cell-free DNA methylation fingerprint for the early detection of gastrointestinal cancers.* Clin Cancer Res
  2021;27(22):6135-6144 **[V]**. Assay: Roche **SeqCap Epi targeted bisulfite** panel (67,832 DMRs, ~25.6 Mb) chosen
  from TCGA/GEO tissue data; ~300 plasma cfDNA samples (healthy, CRC, ESCC/EAC, GC, HCC, PDAC). Plasma came from
  several institutions/countries, so **cancer type and collection site are likely partially aliased** (see Sec. 3e).
  Panel regions were selected *for these cancers*, so every arm starts from a supervised locus preselection.
- **Stackpole et al. 2022** (Nat Commun 13:5566) **[V]** is cfMethyl-Seq (MspI-based enrichment of CpG-rich regions,
  ~3% of genome), 408 plasma samples (colon, liver, lung, stomach + 191 non-cancer); data are in EGA
  (EGAS00001006020), not GEO. It is the same UCLA/UCI group (Zhou XJ, Li W) as CancerLocator/CancerDetector and is
  still the most relevant *method* precedent for read-level features on targeted bisulfite data.
- **HRA003209** (GSA-Human, PRJCA012255, controlled access) = **MONITOR** study, Bie F, Wang Z, Li Y, et al.
  *Multimodal analysis of cell-free DNA whole-methylome sequencing for cancer detection and localization.* Nat Commun
  2023;14:6042 **[V]** (platform "THEMIS"). EM-seq low-pass whole methylome (~2x haploid in the preprint), 780 patients
  with 7 cancers (breast, CRC, esophageal, gastric, liver, lung, pancreas) + 497 controls, six hospitals. Published
  methylation feature: methylated-fragment ratio (fraction of fragments with >=3 CpGs that are fully methylated) in
  1-Mb windows; SVM/LR per modality + elastic-net stacking; RF for tissue of origin.

## 1. Literature tables

Columns: Input | Prediction unit | Supervision | Architecture | Reads->sample aggregation | Advantages |
Disadvantages | Compute | Compatibility with frozen CpG embeddings (Compat.) | Confounding/leakage risk (Risk).

### 1a. cfDNA methylation classifiers (targeted / genome-wide)

| Paper | Year | Input | Unit | Supervision | Architecture | Aggregation | Advantages | Disadvantages | Compute | Compat. | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CancerLocator, Kang et al., Genome Biol 18:53 **[K]** | 2017 | Region-level beta values of cfDNA WGBS at TCGA-derived markers | Sample (tumour fraction + tissue) | Reference tissue profiles (TCGA 450K), no plasma training labels | Probabilistic mixture model (beta distributions), MLE | Region betas -> joint likelihood of tissue type and fraction | Interpretable, yields fraction | Ignores read-level co-methylation; needs tissue references | Negligible | Low: no learned locus features | Marker selection on tissue data only (low) |
| CancerDetector, Li W et al., Nucleic Acids Res 46:e89 **[K]** | 2018 | Individual bisulfite reads at markers | Read likelihood -> sample tumour burden | Reference tumour/normal profiles | Per-read likelihood under tumour vs normal beta profiles; EM for fraction | Probabilistic mixture over all reads | Read resolution; sensitive at low fraction; principled | Assumes CpG independence within read; binary tumour/normal | Negligible | Low | Low (no plasma labels in the model) |
| Stackpole et al., Nat Commun 13:5566 **[V]** | 2022 | cfMethyl-Seq reads in CpG-rich regions | Sample | Sample labels (plasma) + tissue RRBS for markers | Read alpha-value (fraction methylated CpGs per read) thresholds learned per marker (alpha_hyper, alpha_hypo); 4 marker families; linear SVM (L2, C=1) per family, RF (2000 trees) stacker | Counts of hyper/hypo reads per marker, normalised by depth, log -> feature vector | Read-level purification of tumour signal; cross-batch validation reported | Hand-crafted scalar per read (alpha) discards which CpGs are methylated | Low | Low as-is; the alpha statistic is what a set encoder must at least recover | Depth normalisation needed; marker discovery must stay inside training folds |
| EpiPanGI Dx, Kandimalla et al., Clin Cancer Res 27:6135 **[V]** | 2021 | Targeted bisulfite (SeqCap Epi) region methylation | Sample | Sample labels | DMR panels + standard ML classifiers | Region-level methylation | Source of GSE149438; tissue-driven panel | Small n per class; multi-site sample origin | Low | n/a (dataset source) | Site/class aliasing (Sec. 3e) |
| GRAIL CCGA, Liu MC et al., Ann Oncol 31:745 **[K]**, details **[U]** | 2020 | Targeted methylation (bisulfite) panel reads | Sample (cancer + CSO) | Sample labels | Per-fragment abnormality score vs non-cancer methylation model; region-level counts of anomalous fragments; linear/ensemble classifier for detection and tissue-of-origin | Counting fragments whose pattern is unlikely under the healthy model | Scales to ~10^3-10^4 training samples; read-level "witness" logic | Two-stage, not end-to-end; exact fragment model described only in supplement | Moderate (panel), cheap classifier | Medium: a learned per-read score could replace the hand model | Large, well-matched cohort mitigates; CCGA reports depth/site checks |
| MONITOR/THEMIS, Bie et al., Nat Commun 14:6042 **[V]** | 2023 | EM-seq whole methylome: methylation, fragmentation, CNA | Sample | Sample labels | Per-modality SVM/LR, elastic-net stack; RF for TOO | Fraction of fully methylated fragments per 1-Mb window | Multimodal, large cohort, 6 sites | Coarse windows; no learned read model | Low | Low as-is | Six hospitals: site effects; fragment length is a pre-analytic confounder |

### 1b. Read-level deep learning (per-read labels from pure sources)

| Paper | Year | Input | Unit | Supervision | Architecture | Aggregation | Advantages | Disadvantages | Compute | Compat. | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DISMIR, Li J et al., Brief Bioinform 22(6):bbab250 **[V]** | 2021 | WGBS read: one-hot DNA sequence + methylation state, in "switching regions" | Read -> sample risk | Per-read labels by *source*: reads from HCC tumour tissue vs healthy plasma | Deep network (conv/recurrent) on sequence+methylation matrix **[U: exact layers]** | Posterior per read -> likelihood-based sample score | Works at 0.01-0.1x; robust to single-CpG noise | Labels are source labels, not tumour-derived-ness; binary only | Moderate | High: replace one-hot sequence by frozen locus tokens | Tissue vs plasma source label conflates tissue-of-origin with pre-analytics |
| MethylBERT, Jeong Y et al., Nat Commun 16:788 **[V]** | 2025 | Read: 3-mer DNA tokens + per-CpG methylation | Read -> tumour fraction | Pretraining on reference methylomes; fine-tune on reads from pure tumour vs normal references | BERT (pretrained, ~10^8 params), read classification head | Read posteriors -> MLE tumour fraction with prior correction | Strong read classifier; context aware | Big model; needs pure-label reads; binary; fine-tunes sequence model (violates our frozen invariant) | High | Conceptually: frozen locus token replaces 3-mer embedding | Low if references are independent from test plasma |
| Syto, Rizdvanetskyi, Roos, Lutsik, arXiv 2607.04987 **[U: listed as NeurIPS 2026]** | 2026 | Reads from a 39 cell-type atlas | Read -> cell-type fractions | *Soft* per-read labels estimating P(cell type | read pattern) | Read classifier with data-driven soft labels | Deconvolution from read posteriors | Explicitly shows hard per-read labels fail when patterns are shared across many cell types | Atlas-scale | Medium | Supports Sec. 3a: hard pseudo-labels are wrong targets |

### 1c. Atlas / deconvolution methods

| Paper | Year | Input | Unit | Supervision | Architecture | Aggregation | Advantages | Disadvantages | Compute | Compat. | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Moss et al., Nat Commun 9:5068 **[K]** | 2018 | Array (450K/EPIC) betas, cfDNA and purified cell types | Sample fractions | Reference atlas | NNLS on cell-type-specific CpGs | Linear mixture of betas | Simple, interpretable | Not read-level; array resolution | Negligible | None | Low |
| Loyfer et al. (UXM), Nature 613:355 **[K]** | 2023 | WGBS of 39 sorted cell types; fragments with >=3-4 CpGs | Fragment class U/X/M -> sample fractions | Reference atlas | Fragment-level U/X/M calls at cell-type markers; NNLS on U (or M) proportions | Proportion of marker-matching fragments (mean of a per-read indicator) | Fragment-level is markedly more specific than CpG-level; uses co-methylation | Needs sorted-cell references; hard thresholds | Low | Shows mean of per-read features is sufficient for fractions | Low |
| CelFiE, Caggiano et al., Nat Commun 12:2717 **[K]** | 2021 | Per-CpG methylated/unmethylated read counts | Sample fractions (+ unknown components) | Reference + unsupervised components | Bayesian mixture, EM | Counts pooled per site | Handles low coverage, missing references | Discards read co-methylation | Low | None | Low |
| cfSort, Li S et al., **PNAS** 120:e2305236120 **[V]** (not Nat Commun) | 2023 | Fragment-level tissue-specific methylation markers from 521 tissues (29 types) | Sample fractions | Supervised on in-silico mixtures | Marker features -> DNN regressor (ensemble) | Counts of tissue-specific fragments per marker | First supervised cfDNA deconvolution; strong benchmarks | Depends on in-silico mixtures; tissue atlas needed | Moderate | Low (features are hand-made) | Simulation-to-real shift |

### 1d. Set encoders and multiple-instance learning (methodological)

| Paper | Year | Input | Unit | Supervision | Architecture | Aggregation | Advantages | Disadvantages | Compute | Compat. | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deep Sets, Zaheer et al., NeurIPS **[K]** | 2017 | Set of vectors | Set | Set label | rho(sum_i phi(x_i)) | Sum/mean | Universal (with enough latent width), O(n) | No explicit interactions; width must grow with set size for exactness (Wagstaff et al., ICML 2019 **[K]**) | Minimal | High | Sum pooling leaks set size (depth) |
| Set Transformer, Lee J et al., ICML **[K]** | 2019 | Set | Set | Set label | SAB/ISAB self-attention + PMA (k learned seeds) | PMA: attention from learned queries | Models pairwise interactions; ISAB O(nm) | SAB O(n^2) infeasible for 10^6 instances; more params | Low for n<=30, high for bags | High | PMA with many seeds can overfit small n |
| Perceiver, Jaegle et al., ICML **[K]** | 2021 | Large input arrays | Input | Label | Cross-attention from a small learned latent array, then latent self-attention | Learned-query cross-attention (= multi-seed PMA) | Linear in n | Iterative latent blocks are heavy for a few hundred labels | Moderate | High | As PMA |
| Attention MIL, Ilse, Tomczak, Welling, ICML **[K]** | 2018 | Bag of instance embeddings | Bag | Bag label | a_i = softmax(w^T(tanh(Vh_i) * sigmoid(Uh_i))); z = sum a_i h_i | Gated attention weighted mean | Size-invariant (weights sum to 1), interpretable weights, O(n), exact streaming possible | Single softmax; can collapse onto few instances | Minimal | High | Attention can lock onto artefact instances (coverage, locus identity) |
| CLAM, Lu MY et al., Nat Biomed Eng 5:555 **[K]** | 2021 | Frozen patch features, 10^4-10^5 per slide | Slide | Slide label | Gated attention with one attention branch per class + instance clustering loss on top/bottom-k pseudo-labels | Class-specific attention pooling | Designed for frozen features + weak labels + small cohorts | Instance loss assumes top-attended = positive (pseudo-labels) | Low | High (frozen-feature analogue) | Pseudo-label loss amplifies spurious instances |
| TransMIL, Shao et al., NeurIPS **[K]** | 2021 | Patch features | Slide | Slide label | Nystrom self-attention across instances + PPEG positional conv | CLS token | Instance correlations | Needs 2-D layout (PPEG) meaningless for fragments; O(n) only approximately; overfits small cohorts | High for 10^5+ | Medium | Positional module could encode coordinate identity |
| Top-k MIL, Campanella et al., Nat Med 25:1301 **[K]** | 2019 | Tiles | Slide | Slide label | Instance classifier, train on top-k scored instances, then RNN | Max / top-k | Ideal for "any witness => positive" | k and bag size coupled; noisy with small n; binary logic | Moderate | Medium | Selects extreme instances, sensitive to artefacts |
| DSMIL, Li B, Li Y, Eliceiri, CVPR **[K]** | 2021 | Patch features | Slide | Slide label | Max-instance critical score + attention to critical instance | Dual-stream | Robust to sparse positives | Max-instance selection noisy; mostly binary | Low | High | As top-k |
| DTFD-MIL, Zhang H et al., CVPR **[K]** | 2022 | Patch features | Slide | Slide label | Random pseudo-bags per slide, two-tier attention | Pseudo-bag distillation | Formalises multiple sub-bags per sample to enlarge small cohorts | Pseudo-bags inherit bag label (fine when positives are dense, weak for 1% witnesses) | Low | High | Pseudo-bag label noise at low tumour fraction |
| Online softmax, Milakov & Gimelshein, arXiv:1805.02867 **[K]**; FlashAttention, Dao et al., NeurIPS **[K]** | 2018/2022 | - | - | - | Running max + running sum for exact chunked softmax | Exact streaming attention pooling | Exact attention over 10^6+ instances in O(chunk) memory | - | O(n), tiny memory | n/a | none |

### 1e. Sample-as-bag-of-sequences models (weakly supervised, closest precedents)

| Paper | Year | Input | Unit | Supervision | Architecture | Aggregation | Advantages | Disadvantages | Compute | Compat. | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepRC, Widrich et al., NeurIPS **[K]** | 2020 | Immune repertoire: 10^4-10^5+ CDR3 sequences per person | Person | Person label only | 1-D CNN sequence embedding + attention (modern-Hopfield view) with a fixed learned query | Softmax attention over whole repertoire; random subsampling of sequences in training | Designed for witness rates << 1%; same statistical structure as cfDNA | Hopfield framing adds little over gated attention in practice | Moderate | High | Low-witness settings are prone to spurious batch signals |
| DECIDIA, Liu J et al., Mol Oncol **[V]** | 2024 | Raw bisulfite-converted cfDNA fragment sequences (no explicit locus) | Sample | Sample label | Transformer fragment representation + weakly supervised MIL | MIL pooling over sampled fragments | 5,389 samples; multi-cancer; external HCC test | Implicit locus identity through sequence; architecture details paywalled **[U]** | High | Medium | Sequence content can encode coordinates |
| Fragmentia-AI, Xu Y, Bao H et al., Cell Rep Med **[V]** | 2026 | cfDNA fragment sequences (BPE tokens), ultra-low depth | Sample | Sample label (cancer vs non-cancer) | Transformer fragment LM + attention-based MIL pooling | Attention over randomly sampled fragments | 17 cancer types; low depth | Not methylation | High | Medium | Fragmentomics highly sensitive to pre-analytics |
| FLDL, Widrich M et al., reported as ICML 2026 **[U]** | 2026 | Fragments: sequence, methylation, position, strand, HyenaDNA embeddings | Sample (CRC) | Sample label | Fragment encoder + modern-Hopfield attention with learned prototypes | Prototype attention over millions of fragments | 4,394 training samples; explicit artefact controls and dilution series | Not independently verified; large cohort (not ours) | High | High (uses frozen DNA-LM embeddings as one input) | Explicit concern about technical artefacts |

## 2. What the literature implies for this repo

1. Every successful read-level cfDNA method either (i) learns per-read posteriors from **pure** references
   (CancerDetector, DISMIR, MethylBERT, UXM, cfSort) and then aggregates by mixture logic, or (ii) trains end-to-end
   on **sample labels** with a MIL pooling (DeepRC, DECIDIA, Fragmentia, FLDL). Our constraint (plasma, sample labels,
   no deconvolution in the main pipeline) puts us in (ii).
2. The dominant signal is **within-fragment co-methylation at specific loci** (alpha-value, UXM, methylated-fragment
   ratio). The fragment encoder must at least express "fraction of methylated CpGs, conditioned on locus type".
3. Mixture logic says the sample embedding should be **linear in source fractions**: mean pooling of per-fragment
   features does exactly this (sum_s f_s mu_s); softmax attention is a *reweighted* mean that can upweight rare
   informative fragments. Both are bag-size invariant; sum pooling and bag-size features are not.
4. With a few hundred patients, the comparable WSI regime (CLAM on frozen features) uses ~10^5-10^6 trainable
   parameters, heavy dropout and gated attention; transformer-over-instances (TransMIL) is not justified.

## 3. Synthesis

### (a) Per-read pseudo-labelling is not valid here
Assigning the sample label to every read creates targets that are wrong for most reads: at tumour fraction f
(often <1% in early stage, rarely >30%), a fraction 1-f of "cancer" reads are haematopoietic cfDNA indistinguishable
from healthy reads. The Bayes-optimal per-read predictor under this objective is P(sample class | read), which
rewards features **shared by all reads of a sample** (batch/site, conversion efficiency, capture profile, which loci
are covered, fragment length) over features present in the few tumour-derived reads. Consequences:
(1) the model is pushed toward confounders; (2) "per-read accuracy" is meaningless (upper bound ~ f for signal);
(3) averaging read posteriors is simply MIL with mean-of-instance-scores (mi-Net), the weakest MIL variant, trained
with a mis-specified loss; (4) for a representation benchmark it specifically rewards embeddings that make locus
identity easy to memorise (a random per-locus vector is a perfect locus ID). Per-read labels are valid only when
reads come from pure sources (tissue, sorted cells; DISMIR, MethylBERT, UXM) and, even then, shared patterns call for
soft labels (Syto). We therefore train only with the sample-level loss; any instance-level loss (CLAM clustering,
top-k pseudo-labels) is excluded from the primary model.

### (b) Fragment encoder: Deep Sets vs tiny Transformer for 3-30 CpGs
- Deep Sets rho(mean phi(x_i)) with phi(locus, state, geometry) can represent alpha-like statistics and
  locus-conditioned methylation counts; pairwise effects (e.g. "methylated shore CpG with unmethylated island CpG")
  need a latent width that grows with set size (Wagstaff 2019), i.e. implicit, not guaranteed at d=64.
- One self-attention block (SAB) over <=30 tokens represents pairwise locus x state interactions explicitly, which is
  exactly where a functional locus embedding could add information beyond position (co-methylation of CpGs with
  shared regulatory context). Cost: 30^2 x d per fragment, negligible compared with the K fragments per bag.
- Risk of the Transformer: relative-position/gap tokens make it easy to model local geometry independent of the
  locus representation. Mitigation: geometry enters as a small additive embedding identical across arms; run the
  `position_only` and `random` arms through the same encoder.
- Decision: **1-layer SAB + PMA(1 seed) fragment encoder as primary**; Deep Sets (mean+max pooled phi) as the
  fragment-encoder ablation. Both have near-identical parameter counts at d=64.

### (c) Sample aggregator for 10^4-10^6 fragments
- **Mean pooling**: mixture-consistent, cannot collapse, cheapest; dilutes rare witnesses (effect size ~ f).
- **Gated attention MIL** (Ilse; CLAM multi-branch): weighted mean, still size-invariant, can up-weight witnesses;
  exact for any N via online softmax (running max m, running sum s = sum e^(a_i - m), running weighted sum
  z = sum e^(a_i - m) h_i, merged across chunks). One attention branch per class matches tissue-of-origin, where
  each class has its own marker fragments.
- **PMA** (k seeds, multi-head): generalises gated attention with multiple queries; more parameters, and the
  multi-head/multi-seed capacity is the first to overfit at n~300. Also exact-streamable per head.
- Excluded: TransMIL/SAB across instances (O(N^2) or approximations; no meaningful instance order), top-k/max
  (bag-size dependent, brittle at small n, and bags of different depth are not comparable).
- Memory/compute: train on random bags of K fragments per sample (DeepRC and Fragmentia precedent; DTFD pseudo-bags),
  several bags per sample per epoch; evaluate exactly on all fragments (or a fixed seeded cap) in chunks. Train/test
  bag-size mismatch is benign for softmax/mean pooling (both are normalised averages); verify by reporting test
  metrics at K_test in {K_train, 4K_train, all}.

### (d) Parameter parity across arms
Only the input projection depends on the representation dimension D (256 functional, >=1k for sequence LMs, 0 for
methylation-only). Options, in order of preference:
1. **Label-free PCA to a common r** (default r=64) fitted per representation on a seeded random sample of genome-wide
   CpGs from the embedding store (no patient data), whitened; then a shared trainable `Linear(r, d_model) + LayerNorm`.
   Trainable parameter counts are identical across all arms with a locus vector. Cost: loses variance beyond r;
   report retained variance per arm and run r=128 as sensitivity.
2. Trainable `Linear(D, d_model)` on the full embedding (report D x d_model extra params per arm) as sensitivity:
   if conclusions flip between 1 and 2, the effect is about capacity, not representation.
3. Fixed random orthogonal projection to r (JL): parity without fitting, but noisier than PCA.
- **methylation_only**: locus vector = one learned constant token (r-dim) passed through the same adapter, so the
  model sees state + geometry only. It is impossible to equalise *effective* capacity here (the representation *is*
  the capacity); declare trainable counts, and rely on controls:
  - `random` (hash Gaussian, same r): unique locus IDs with no biology -> measures coordinate memorisation capacity;
  - `shuffled_functional` (permute embedding rows across loci; recommended addition): same marginal distribution,
    destroyed locus-function mapping;
  - `position_only`. A functional gain is claimed only if functional > random and > shuffled, paired over patients.

### (e) Confounders and leakage
- **Depth/coverage**: fragment counts differ by class and batch. Fixed-K bags and normalised pooling remove N from
  the input; never use sum pooling, fragment counts, or coverage as features. Coverage *composition* (which loci are
  captured) still differs by batch: add a **coverage-only control** (states replaced by a constant or permuted within
  each sample, loci kept). High accuracy there quantifies how much the task is solvable without methylation.
- **Coordinate memorisation**: with a targeted panel, "fragment at locus X" is informative only via selection or
  capture efficiency. Controls above (random, shuffled, coverage-only) measure it; optional region-balanced bag
  sampling (equal fragments per panel region) removes the capture-profile signal at the cost of some sensitivity.
- **Site/cohort**: GSE149438 cancer types were collected at different centres; MONITOR has six hospitals. Report a
  batch-predictability probe (same model predicting site from the bag), per-site metrics, and leave-site-out where
  each class spans >=2 sites. Conversion efficiency and global methylation level differ by batch and also reach the
  methylation-only arm; report per-sample global methylation by class/site.
- **Pre-analytics**: do not input fragment length or end motifs (fragmentomics confounds with collection protocol).
  Autosomes only (sex). Fixed minimum CpGs per fragment (>=3) identical across arms.
- **Leakage**: patient-level splits (already specified); any supervised region/marker selection only inside training
  folds; normalisation statistics from the representation store, not from patients; model selection on validation
  only; test evaluated once.

## 4. PRIMARY ARCHITECTURE RECOMMENDATION

**Fragment encoder: 1-layer set-attention (SAB) + PMA-1; sample aggregator: multi-branch gated-attention MIL
(Ilse 2018 / CLAM 2021), trained end-to-end on sample labels with random fixed-size fragment bags; exact streaming
attention at test time.** Motivation: MIL with normalised attention is the established approach when only bag labels
exist and witnesses are sparse (Ilse, CLAM, DeepRC, DECIDIA, Fragmentia); per-fragment co-methylation is the proven
unit of cfDNA signal (alpha-value, UXM, MFR) and a single attention layer models it explicitly at negligible cost;
size-invariant pooling protects against depth confounding; small width follows the frozen-feature small-cohort
regime of CLAM.

Token and fragment encoder
- Token: `LN(Linear(r=64 -> d=64)(PCA_r(locus)))` + `Emb(state in {0,1}, d)` + `MLP(log1p gap to previous CpG,
  log1p offset from first CpG, rank/L) -> d`. Token dropout p=0.1 (drop CpGs, keep >=3).
- 1 pre-LN SAB block: 4 heads, d=64, FFN 128, dropout 0.1. PMA with 1 learned seed -> fragment embedding h in R^64.
- Fragments with <3 CpGs dropped; cap 32 CpGs (truncate by position; report fraction truncated).

Sample aggregator (primary)
- Gated attention, hidden 32, one attention branch per class c: a_ic = w_c^T(tanh(V h_i) * sigmoid(U h_i)),
  z_c = sum_i softmax_i(a_ic) h_i; logit_c = <head_c, dropout_0.25(LN(z_c))>. Attention dropout on instances 0.1.
- Total trainable parameters ~6-8 x 10^4 (identical across arms under PCA-r parity).

Ablation aggregators
1. **Mean pooling** (same fragment encoder; mixture-consistent baseline). If attention does not beat mean, report
   mean as the conservative result.
2. **PMA** with 1 seed x 4 heads (Set Transformer pooling), streamed exactly per head.
Fragment-encoder ablation: Deep Sets (phi = 2-layer MLP d=64, mean+max pooling).

Optimisation and regularisation (n ~ 300 patients)
- AdamW, lr 5e-4, weight decay 0.05, cosine decay, 3 % warm-up, grad-clip 1.0; label smoothing 0.1;
  class-balanced sampling (or inverse-frequency loss weights) with macro-F1 / balanced accuracy for selection.
- Step = B=8 samples x K=4096 fragments (K in {2048, 8192} as sensitivity). Epoch = every training sample seen in
  M=4 independent random bags. Max 100 epochs; early stopping on validation macro-F1 (patience 15), best checkpoint
  reloaded. 5 training seeds x outer CV folds for GSE149438; paper split for HRA003209.
- Bag sampler draws fragments uniformly without replacement (with replacement if N<K); optional region-balanced mode
  for the confounding control. Same sampler seeds for every arm (paired comparison).

Evaluation (deterministic)
- Primary: stream all fragments of each sample in chunks of 65,536 through the fragment encoder and merge
  per-branch online-softmax accumulators (exact, O(chunk) memory). For HRA003209 at ~2x, if full streaming is too
  slow, use a fixed seeded subset (e.g. 10^6 fragments) identical across arms.
- Secondary check: mean of logits over 16 fixed seeded bags of K_train; report agreement with full streaming.
- Paired patient-level bootstrap/permutation across arms (as in REFACTOR_PLAN).

What to cache
- Per dataset: CSR fragment store (offsets, locus_key, state) and the sorted locus universe.
- Per representation: `PCA_r(embedding)` rows for the dataset universe (fp16) + PCA manifest (seed, retained
  variance); targeted panel fits on GPU, whole-methylome universe as CPU memmap gathered per batch.
- Not cacheable during training: fragment embeddings (the encoder trains). After training, cache per-fragment
  embeddings and attention scores of the final model for interpretation (top-attended loci per class) and for the
  batch-predictability probe; never reuse them to train another arm.
