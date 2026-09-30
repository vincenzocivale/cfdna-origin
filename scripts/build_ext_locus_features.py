#!/usr/bin/env python3
"""Compute functional features for CpGs missing from the benchmark's locus_features_v1 store.

Uses the *same* code and frozen sources (annotation_core_v1 contract, GENCODE v50 / UCSC islands / cCRE V4,
4165 ENCODE peak tracks) as the original store, via the MehylPredictor builder, so features are identical by
construction. Universe: every CpG of hg38 chr1-22,X,Y (wgbstools index). chrM is not supported by the builder's
locus-key scheme and is excluded (reported in the manifest).

  --regress N   recompute N random already-stored chr21 loci and require exact equality with the store
  --build       compute all missing loci per chromosome into data/ext_store/features/<chrom>/
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

MP = Path("/data2/home/vcivale/projects/methylation/MehylPredictor")
sys.path.insert(0, str(MP / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfdna_too.data.cpg_index import CHROMS, CpGIndex  # noqa: E402
from methylation_predictor.locus_features.annotations import (  # noqa: E402
    ReferenceAnnotationEngine,
    encode_annotation_core,
)
from methylation_predictor.locus_features.regulatory import compute_regulatory_overlaps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "CpGRepresentationBenchmark"
STORE = BENCH / "data/external/functional/locus_features_v1"
EXT = ROOT / "data/ext_store"
CONTRACT = MP / "resources/locus_features/annotation_core_v1.json"
FAI = EXT / "ref/hg38.fa.fai"
KEYCODE = {**{f"chr{i}": i for i in range(1, 23)}, "chrX": 23, "chrY": 24}


def keys_for(chrom: str, pos: np.ndarray) -> np.ndarray:
    return (np.uint64(KEYCODE[chrom]) << np.uint64(32)) | pos.astype(np.uint64)


def compute(chrom: str, pos: np.ndarray):
    engine = ReferenceAnnotationEngine(EXT / "sources", FAI)
    raw = engine.compute_raw_annotations(chrom, pos)
    core = encode_annotation_core(raw, CONTRACT)
    packed, breadth = compute_regulatory_overlaps(
        chrom, pos, STORE / "regulatory/track_contract.tsv", STORE / "regulatory/sources"
    )
    return raw, core, packed, breadth


def build_chrom(chrom: str) -> dict:
    ix = CpGIndex(ROOT / "data/hg38_cpg_index.npz")
    i = CHROMS.index(chrom)
    fa = ix.positions[ix.offsets[i] : ix.offsets[i + 1]].astype(np.int64)
    if chrom in KEYCODE and (STORE / f"catalog/{chrom}.parquet").exists():
        have = pd.read_parquet(STORE / f"catalog/{chrom}.parquet", columns=["position"]).position.to_numpy()
        pos = np.setdiff1d(fa, have)
    else:
        pos = fa
    out = EXT / "features" / chrom
    out.mkdir(parents=True, exist_ok=True)
    if len(pos) == 0:
        (out / "manifest.json").write_text(json.dumps({"chromosome": chrom, "number_of_loci": 0, "status": "complete"}))
        return {"chrom": chrom, "n": 0}
    raw, core, packed, breadth = compute(chrom, pos)
    np.save(out / "position.npy", pos)
    np.save(out / "locus_key.npy", keys_for(chrom, pos))
    np.save(out / "annotation_core.f32.npy", core)
    np.save(out / "breadth.f32.npy", breadth)
    np.save(out / "regulatory.packbits.npy", packed)
    (out / "manifest.json").write_text(json.dumps({"chromosome": chrom, "number_of_loci": int(len(pos)), "status": "complete"}))
    return {"chrom": chrom, "n": int(len(pos))}


def regress(n: int) -> None:
    chrom = "chr21"
    shard = STORE / "features" / chrom
    keys = np.load(shard / "locus_key.npy")
    rows = np.sort(np.random.default_rng(0).choice(len(keys), n, replace=False))
    pos = np.load(shard / "position.npy")[rows].astype(np.int64)
    _, core, packed, breadth = compute(chrom, pos)
    ok_core = np.array_equal(core, np.load(shard / "annotation_core.f32.npy", mmap_mode="r")[rows])
    ok_pack = np.array_equal(packed, np.load(shard / "regulatory.packbits.npy", mmap_mode="r")[rows])
    ok_br = np.array_equal(breadth, np.load(shard / "breadth.f32.npy", mmap_mode="r")[rows])
    print(json.dumps({"rows": n, "annotation_core_equal": ok_core, "packbits_equal": ok_pack, "breadth_equal": ok_br}))
    if not (ok_core and ok_pack and ok_br):
        raise SystemExit("regression against the existing store FAILED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regress", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.regress:
        regress(args.regress)
    if args.build:
        chroms = [c for c in CHROMS if c in KEYCODE]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(build_chrom, c) for c in chroms]
            results = [f.result() for f in as_completed(futs)]
        total = sum(r["n"] for r in results)
        (EXT / "manifest.json").write_text(json.dumps({"loci": total, "chroms": chroms, "excluded": ["chrM"]}, indent=2))
        print("built", total, "loci")


if __name__ == "__main__":
    main()
