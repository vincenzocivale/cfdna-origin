#!/usr/bin/env python3
"""Genome-wide frozen functional locus embedding (unsupervised PCA/SVD), same recipe as the benchmark's
`build_functional_pca_embedding.py`: unit-scaled binary ENCODE track block (4165) + standardized dense block
(18 annotation-core + 5 breadth) -> TruncatedSVD(256).

Universe: the benchmark's locus_features_v1 store (27,078,662 loci) + data/ext_store (2,322,698 loci) = every CpG
of hg38 chr1-22,X,Y (29,401,360). Rows are sorted chr1..22,X,Y then position, which is exactly the wgbstools
global CpG order, so PAT CpG index i (1-based) <-> embedding row i-1 (asserted against the FASTA-derived index).
The SVD is fitted on a seeded random sample of loci from the whole universe; no methylation values, no patients.

Output: /cpg_idx int64 (= locus_key: chrom_code<<32 | 1-based position), /embedding float16 [N,256]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from cpg_index import CHROMS, CpGIndex

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT.parent / "CpGRepresentationBenchmark/data/external/functional/locus_features_v1/features"
EXT = ROOT / "data/ext_store/features"
UCHROMS = [c for c in CHROMS if c != "chrM"]
N_TRACKS, N_BYTES = 4165, 521


def load_chrom(chrom: str, rows: np.ndarray | None = None):
    """Merged (main + ext) shard for a chromosome, position-sorted. rows selects a subset of merged rows."""
    parts = [d for d in (STORE / chrom, EXT / chrom) if (d / "locus_key.npy").exists()]
    keys = [np.load(d / "locus_key.npy") for d in parts]
    order = np.argsort(np.concatenate(keys), kind="stable")
    allkeys = np.concatenate(keys)[order]
    if rows is None:
        rows = np.arange(len(allkeys))
    src = order[rows]  # index into concatenated parts
    bounds = np.cumsum([0] + [len(k) for k in keys])
    ac, br, pk = [], [], []
    for p, d in enumerate(parts):
        sel = src[(src >= bounds[p]) & (src < bounds[p + 1])] - bounds[p]
        ac.append((sel, np.load(d / "annotation_core.f32.npy", mmap_mode="r")))
        br.append(np.load(d / "breadth.f32.npy", mmap_mode="r"))
        pk.append(np.load(d / "regulatory.packbits.npy", mmap_mode="r"))
    dense = np.empty((len(rows), 23), np.float32)
    packed = np.empty((len(rows), N_BYTES), np.uint8)
    part_of = np.searchsorted(bounds, src, side="right") - 1
    for p in range(len(parts)):
        m = part_of == p
        local = src[m] - bounds[p]
        dense[m] = np.hstack([ac[p][1][local], br[p][local]])
        packed[m] = pk[p][local]
    return allkeys[rows], dense, packed


def chrom_sizes() -> dict[str, int]:
    out = {}
    for c in UCHROMS:
        n = sum(len(np.load(d / "locus_key.npy", mmap_mode="r")) for d in (STORE / c, EXT / c) if (d / "locus_key.npy").exists())
        out[c] = n
    return out


def to_sparse(packed: np.ndarray) -> sparse.csr_matrix:
    bits = np.unpackbits(packed, axis=1, bitorder="big")[:, :N_TRACKS]
    return sparse.csr_matrix(bits.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "data/functional_pca_genomewide_hg38.h5")
    ap.add_argument("--fit-loci", type=int, default=1_000_000)
    ap.add_argument("--n-components", type=int, default=256)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sizes = chrom_sizes()
    total = sum(sizes.values())
    print("universe", total, flush=True)
    ix = CpGIndex(ROOT / "data/hg38_cpg_index.npz")
    rng = np.random.default_rng(args.seed)

    # ---- fit on a genome-wide seeded sample
    blocks_t, blocks_d = [], []
    for c in UCHROMS:
        k = int(round(args.fit_loci * sizes[c] / total))
        rows = np.sort(rng.choice(sizes[c], k, replace=False))
        _, d, p = load_chrom(c, rows)
        blocks_t.append(to_sparse(p)); blocks_d.append(d)
        print("sampled", c, k, flush=True)
    tracks = sparse.vstack(blocks_t, format="csr"); dense = np.vstack(blocks_d)
    mean = dense.mean(0, keepdims=True); std = dense.std(0, keepdims=True); std[std == 0] = 1.0
    scale = np.float32(1.0 / max(np.sqrt(tracks.multiply(tracks).mean()), 1e-8))
    combined = sparse.hstack([tracks.multiply(scale), sparse.csr_matrix((dense - mean) / std)], format="csr")
    svd = TruncatedSVD(n_components=args.n_components, random_state=args.seed).fit(combined)
    print("explained variance", float(svd.explained_variance_ratio_.sum()), flush=True)
    comp = torch.tensor(svd.components_.T.astype(np.float32), device=args.device)  # [4188, D]
    w_tracks, w_dense = comp[:N_TRACKS] * float(scale), comp[N_TRACKS:]
    mean_t = torch.tensor(mean, device=args.device); std_t = torch.tensor(std, device=args.device)
    del combined, tracks, dense, blocks_t, blocks_d

    # ---- project every locus
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row0 = 0
    with h5py.File(args.output, "w") as out:
        d_key = out.create_dataset("cpg_idx", (total,), dtype="int64")
        d_emb = out.create_dataset("embedding", (total, args.n_components), dtype="float16", chunks=(65536, args.n_components))
        for c in UCHROMS:
            keys, dn, pk = load_chrom(c)
            i = CHROMS.index(c)
            fa_pos = ix.positions[ix.offsets[i] : ix.offsets[i + 1]]
            if not np.array_equal(keys & np.uint64(0xFFFFFFFF), fa_pos.astype(np.uint64)):
                raise RuntimeError(f"{c}: merged store does not equal the FASTA CpG order")
            if ix.offsets[i] != row0:
                raise RuntimeError(f"{c}: row offset {row0} != global CpG offset {ix.offsets[i]}")
            for s in range(0, len(keys), 100_000):
                bits = torch.from_numpy(np.unpackbits(pk[s : s + 100_000], axis=1, bitorder="big")[:, :N_TRACKS]).to(args.device)
                x = bits.float() @ w_tracks + ((torch.from_numpy(dn[s : s + 100_000]).to(args.device) - mean_t) / std_t) @ w_dense
                d_emb[row0 + s : row0 + s + len(bits)] = x.half().cpu().numpy()
            d_key[row0 : row0 + len(keys)] = keys.astype(np.int64)
            row0 += len(keys)
            print("projected", c, len(keys), flush=True)
        out.attrs.update(
            representation="functional_annotations_pca_genomewide", n_components=args.n_components,
            explained_variance_ratio_sum=float(svd.explained_variance_ratio_.sum()), fit_loci=args.fit_loci, fit_seed=args.seed,
            created_utc=datetime.now(timezone.utc).isoformat(), patient_specific=False, supervision="none",
            coordinate_convention="1-based position of CpG cytosine", cpg_namespace="locus_key_chromcode<<32|pos_v1",
            reference_build="GRCh38", row_order="wgbstools_global_cpg_index_chr1-22XY", excluded="chrM",
        )
    np.savez_compressed(args.output.with_suffix(".projection.npz"), dense_mean=mean, dense_std=std, track_scale=scale, components=svd.components_.astype(np.float32))
    args.output.with_suffix(".h5.json").write_text(json.dumps({"n_loci": total, "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()), "top10": svd.explained_variance_ratio_[:10].tolist()}, indent=2))
    print("done", row0)


if __name__ == "__main__":
    main()
