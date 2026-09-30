"""HRA003209 / PRJCA012255 — MONITOR cohort (Bie et al., Nat Commun 2023;14:6042). CONTROLLED ACCESS.

1277 plasma cfDNA samples (one per participant), low-pass whole-methylome EM-seq, PE100; Bismark 0.19 -> hg19 BAMs.
Metadata sources (both public, no data access needed):
  - paper Supplementary Data 4 (MOESM5_ESM.xlsx): Patient ID, Age bin, Sex, Diagnosis, Cancer stage, Group
    (Training/Test = official split), Hospital source (1-6). The same sheet contains the paper's model scores
    (MFR, FSI, CAFF, FEM, THEMIS): they are deliberately NOT carried into the sample table.
  - GSA public metadata (HRA003209_public_metadata.xlsx, "Individuals & samples" + "Files"): patient -> BAM run.
Expected raw layout once access is granted: <raw_dir>/<bam_run>.bam (e.g. HRR775039.bam).
Known confounders: hospital is aliased with class (breast only from hospital 1; healthy only from hospitals 2-3);
controls are younger; depth varies.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SUPP_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-41774-w/MediaObjects/41467_2023_41774_MOESM5_ESM.xlsx"
GSA_BROWSE_URL = "https://ngdc.cncb.ac.cn/gsa-human/browse/HRA003209"
PAPER_SCORE_COLUMNS = ("MFR", "FSI", "CAFF", "FEM", "THEMIS")  # model outputs of the original paper: never features


def build_sample_table(supp_xlsx: Path, gsa_xlsx: Path) -> pd.DataFrame:
    supp = pd.read_excel(supp_xlsx, sheet_name="Supplementary Data 4", header=1)
    supp.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in supp.columns]
    supp = supp[[c for c in supp.columns if not c.startswith(PAPER_SCORE_COLUMNS)]]
    ind = pd.read_excel(gsa_xlsx, sheet_name="Individuals & samples")
    files = pd.read_excel(gsa_xlsx, sheet_name="Files")
    bam = files[files["Files"].astype(str).str.contains(r"\.bam", regex=True)]
    bam = bam.assign(bam_bytes=bam["Files"].str.extract(r"\((\d+) bytes\)")[0].astype("int64"))
    t = ind.merge(bam[["Sample", "Run accession", "bam_bytes"]], left_on="Sample Accession", right_on="Sample",
                  how="left", validate="one_to_one")
    df = supp.merge(t, left_on="Patient ID", right_on="Individual Identifier", how="outer", indicator=True,
                    validate="one_to_one")
    if (df._merge != "both").any():
        raise ValueError(f"paper/GSA patient mismatch: {df.loc[df._merge != 'both', ['Patient ID', 'Individual Identifier']].head()}")
    split = df["Group"].map({"Training": "train", "Test": "test"})
    if split.isna().any():
        raise ValueError(f"unexpected Group values: {sorted(df.Group.unique())}")
    return pd.DataFrame({
        "sample_id": df["Sample Accession"], "patient_id": df["Patient ID"], "diagnosis": df["Diagnosis"],
        "sample_type": "plasma_cfdna", "official_split": split, "batch": "hospital_" + df["Hospital source"].astype(str),
        "sex": df["Sex"], "age": df["Age"], "stage": df["Cancer stage"],
        "source_accession": df["Run accession"], "bam_bytes": df["bam_bytes"],
    })


def expected_bam(raw_dir: Path, row) -> Path:
    return raw_dir / f"{row.source_accession}.bam"


def missing_inputs(meta: pd.DataFrame, raw_dir: Path) -> list[str]:
    return [str(expected_bam(raw_dir, r)) for r in meta.itertuples() if not expected_bam(raw_dir, r).exists()]
