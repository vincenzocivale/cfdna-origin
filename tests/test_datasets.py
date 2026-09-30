import gzip
import shlex

import openpyxl
import pandas as pd
import pytest

from cfdna_origin.config import load_component
from cfdna_origin.data.datasets import gse149438, hra003209

SOFT = """\
^SERIES = GSE149438
!Series_title = EpiPanGI Dx
^SAMPLE = GSM1
!Sample_title = KRp1-HCC_43
!Sample_source_name_ch1 = plasma
!Sample_characteristics_ch1 = tissue: HCC
!Sample_characteristics_ch1 = alternate id: HCC_43
!Sample_characteristics_ch1 = age: 61
^SAMPLE = GSM2
!Sample_title = KRp2-CRC_7
!Sample_source_name_ch1 = plasma
!Sample_characteristics_ch1 = tissue: CRC
!Sample_characteristics_ch1 = alternate id: CRC_7
!Sample_characteristics_ch1 = age: unknown
"""
ENA = pd.DataFrame({
    "run_accession": ["SRR1", "SRR2"], "sample_alias": ["GSM1", "GSM2"],
    "fastq_ftp": ["ftp.sra/SRR1_1.fastq.gz;ftp.sra/SRR1_2.fastq.gz", "ftp.sra/SRR2_1.fastq.gz;ftp.sra/SRR2_2.fastq.gz"],
    "fastq_md5": ["aa;bb", "cc;dd"], "read_count": [100, 200], "fastq_bytes": ["10;20", "30;40"],
})


def _write_soft(path, text=SOFT):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


@pytest.fixture()
def geo_files(tmp_path):
    ENA.to_csv(tmp_path / "ena.tsv", sep="\t", index=False)
    return _write_soft(tmp_path / "family.soft.gz"), tmp_path / "ena.tsv"


def test_parse_soft(geo_files):
    soft = gse149438.parse_soft(geo_files[0])
    assert soft.gsm.tolist() == ["GSM1", "GSM2"]
    assert soft.char_alternate_id.tolist() == ["HCC_43", "CRC_7"] and soft.char_tissue.tolist() == ["HCC", "CRC"]


def test_build_sample_table(geo_files):
    t = gse149438.build_sample_table(*geo_files)
    for col in ("sample_id", "patient_id", "diagnosis", "sample_type", "batch", "source_accession", "fastq_ftp"):
        assert col in t
    assert t.patient_id.tolist() == ["HCC_43", "CRC_7"] and t.batch.tolist() == ["KRp1", "KRp2"]
    assert t.diagnosis.tolist() == ["HCC", "CRC"] and t.source_accession.tolist() == ["SRR1", "SRR2"]
    assert t.fastq_bytes.tolist() == [30, 70] and t.age.iloc[0] == 61 and pd.isna(t.age.iloc[1])


def test_build_sample_table_errors(tmp_path, geo_files):
    ENA.iloc[:1].to_csv(tmp_path / "ena1.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="without ENA runs"):
        gse149438.build_sample_table(geo_files[0], tmp_path / "ena1.tsv")
    same_patient = _write_soft(tmp_path / "dup.soft.gz", SOFT.replace("alternate id: CRC_7", "alternate id: HCC_43"))
    with pytest.raises(ValueError, match="one sample per patient"):
        gse149438.build_sample_table(same_patient, geo_files[1])
    dup_gsm = _write_soft(tmp_path / "dupgsm.soft.gz", SOFT.replace("^SAMPLE = GSM2", "^SAMPLE = GSM1"))
    with pytest.raises(pd.errors.MergeError):  # one_to_one merge with the ENA runs
        gse149438.build_sample_table(dup_gsm, geo_files[1])


def test_sample_commands(tmp_path, geo_files):
    row = gse149438.build_sample_table(*geo_files).iloc[0]
    pp = load_component("datasets", "gse149438")["preprocessing"]
    cmds = gse149438.sample_commands(row, raw_dir=tmp_path / "raw", bismark_index=tmp_path / "idx", pp=pp, threads=16)
    text = "\n".join(cmds)
    assert sum(c.startswith("aria2c") for c in cmds) == 2 and sum("md5sum -c" in c for c in cmds) == 2
    assert "https://ftp.sra/SRR1_1.fastq.gz" in text and "aa  " in text
    t = pp["trim"]
    trim = next(c for c in cmds if c.startswith("trim_galore"))
    for flag in ("clip_R1", "clip_R2", "three_prime_clip_R1", "three_prime_clip_R2"):
        assert f"--{flag} {t[flag.lower()]} " in trim
    assert "--cores 8" in trim
    assert any(c.startswith("bismark ") and f"--parallel {pp['bismark']['parallel']}" in c for c in cmds)
    assert any(c.startswith("deduplicate_bismark") for c in cmds)
    assert cmds[-1].endswith(shlex.quote(str(tmp_path / "raw" / "GSM1" / "GSM1.dedup.bam")))
    with pytest.raises(ValueError, match="2 paired"):
        gse149438.sample_commands(row.copy().replace(row.fastq_ftp, "ftp/one.fq.gz"), raw_dir=tmp_path,
                                  bismark_index=tmp_path, pp=pp, threads=1)


# ---- HRA003209 ------------------------------------------------------------------------------------------------------
SUPP_HEADER = ["Patient ID", "Age", "Sex", "Diagnosis", "Cancer\nstage", "Group", "Hospital source", "MFR",
               "THEMIS score"]
SUPP_ROWS = [["P1", "50-59", "F", "BRCA", "II", "Training", 1, 0.3, 0.9],
             ["P2", "40-49", "M", "HEALTHY", "-", "Test", 2, 0.1, 0.2],
             ["P3", "60-69", "M", "HCC", "III", "Training", 3, 0.5, 0.7]]


def _xlsx(path, sheets):
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)
    return path


@pytest.fixture()
def hra_files(tmp_path):
    supp = _xlsx(tmp_path / "supp.xlsx", {"Supplementary Data 4": [["Supplementary Data 4. Participants"], SUPP_HEADER,
                                                                   *SUPP_ROWS]})
    gsa = _xlsx(tmp_path / "gsa.xlsx", {
        "Individuals & samples": [["Individual Identifier", "Sample Accession"], ["P1", "HRS1"], ["P2", "HRS2"],
                                  ["P3", "HRS3"]],
        "Files": [["Sample", "Run accession", "Files"], ["HRS1", "HRR1", "HRR1.bam (1000 bytes)"],
                  ["HRS1", "HRR1", "HRR1.md5 (32 bytes)"], ["HRS2", "HRR2", "HRR2.bam (2000 bytes)"],
                  ["HRS3", "HRR3", "HRR3.bam (3000 bytes)"]],
    })
    return supp, gsa


def test_hra003209_sample_table(tmp_path, hra_files):
    t = hra003209.build_sample_table(*hra_files)
    assert t.sample_id.tolist() == ["HRS1", "HRS2", "HRS3"] and t.patient_id.tolist() == ["P1", "P2", "P3"]
    assert t.official_split.tolist() == ["train", "test", "train"]
    assert t.batch.tolist() == ["hospital_1", "hospital_2", "hospital_3"]
    assert t.stage.tolist() == ["II", "-", "III"] and t.bam_bytes.tolist() == [1000, 2000, 3000]
    assert t.source_accession.tolist() == ["HRR1", "HRR2", "HRR3"]
    assert not {c for c in t.columns if c.startswith(hra003209.PAPER_SCORE_COLUMNS)}
    ds = load_component("datasets", "hra003209")
    assert set(t.diagnosis) <= set(ds["label_schemes"][ds["label_scheme"]]["map"])
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "HRR2.bam").write_bytes(b"")
    assert hra003209.missing_inputs(t, raw) == [str(raw / "HRR1.bam"), str(raw / "HRR3.bam")]


def test_hra003209_patient_mismatch(tmp_path, hra_files):
    supp = _xlsx(tmp_path / "supp2.xlsx", {"Supplementary Data 4": [["title"], SUPP_HEADER, *SUPP_ROWS[:2]]})
    with pytest.raises(ValueError, match="mismatch"):
        hra003209.build_sample_table(supp, hra_files[1])
