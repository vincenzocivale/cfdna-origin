#!/usr/bin/env python3
"""Parse GSE186458 SOFT metadata into a sample table with cell-type/tissue labels and a donor-proxy group.

GEO gives no donor id (the `Z...` code in file names is the SAMPLE id). Samples from one donor share
lab/age/sex/race, so `donor_group` = (lab, age, sex, race) is used as a leakage-safe proxy: it can only merge
distinct donors (over-grouping), never split one donor across groups. Replace with the true donor id if the
Loyfer supplementary table is added.
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import pandas as pd


def parse(soft: Path) -> pd.DataFrame:
    rows, cur = [], None
    with gzip.open(soft, "rt") as fh:
        for line in fh:
            if line.startswith("^SAMPLE"):
                cur = {"gsm": line.split("=")[1].strip()}
                rows.append(cur)
            elif cur is not None and line.startswith("!Sample_title"):
                cur["title"] = line.split("=", 1)[1].strip()
            elif cur is not None and line.startswith("!Sample_characteristics_ch1"):
                k, _, v = line.split("=", 1)[1].strip().partition(":")
                cur[k.strip().lower()] = v.strip()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--soft", type=Path, required=True)
    ap.add_argument("--pat-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = parse(a.soft)
    files = {re.match(r"(GSM\d+)_", p.name).group(1): p.name for p in a.pat_dir.glob("*.hg38.pat.gz")}
    listing = pd.read_csv(a.pat_dir.parent / "filelist.txt", sep="\t")
    names = {re.match(r"(GSM\d+)_", n).group(1): n for n in listing.Name if n.endswith(".hg38.pat.gz")}
    df = df[df.gsm.isin(names)].copy()
    df["pat_file"] = df.gsm.map(names)
    df["sample_code"] = df.title.str.rsplit("-", n=1).str[1]
    df["cell_type"] = df.title.str.rsplit("-", n=1).str[0]
    df = df.rename(columns={"tissue": "tissue_source"})
    # WBC_<id> / cfDNA_<id> pairs are healthy-plasma cfDNA and matched white blood cells: real cfDNA test data,
    # never part of the pure-cell-type reference used for training.
    df["role"] = "reference"
    df.loc[df.title.str.startswith("WBC_"), "role"] = "wbc"
    df.loc[df.title.str.startswith("cfDNA_"), "role"] = "cfdna"
    df["organ"] = df.cell_type.str.split("-").str[0]
    df["donor_group"] = df[["lab", "age", "sex", "race"]].astype(str).agg("|".join, axis=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(df.role.value_counts().to_dict())
    print(len(df), "samples;", df.cell_type.nunique(), "cell types;", df.tissue_source.nunique(), "tissue sources;",
          df.donor_group.nunique(), "donor groups")
    print(df.groupby("donor_group").size().describe())
    print(df.groupby("cell_type").donor_group.nunique().sort_values().head(10))


if __name__ == "__main__":
    main()
