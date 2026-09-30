"""Train / evaluate one arm. Model selection uses VAL only; TEST and external cfDNA are evaluated once at the end."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from cfdna_too.data.cpg_index import CpGIndex
from cfdna_too.data.readstore import ReadStore
from cfdna_too.models.set_classifier import SetClassifier
from cfdna_too.training.tables import load_table


@torch.no_grad()
def predict_sample(model, store, i, max_reads, bs=8192):
    idx = store.eval_indices(i, max_reads)
    lo, lf = [], []
    for s in range(0, len(idx), bs):
        r, st, m = store.batch(idx[s : s + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, f = model(r, st, m)
        lo.append(o.float().log_softmax(-1)); lf.append(f.float().log_softmax(-1))
    return torch.cat(lo), torch.cat(lf)


def evaluate(model, store, organ_id, fine_id, max_reads, n_organ):
    """Per-read and per-sample (mean log-prob over reads) organ metrics for every sample in `store`."""
    model.eval()
    y, p, ys, ps, rows = [], [], [], [], []
    for i, r in store.samples.iterrows():
        lo, _ = predict_sample(model, store, i, max_reads)
        y.append(np.full(len(lo), organ_id[i])); p.append(lo.argmax(1).cpu().numpy())
        ps.append(int(lo.mean(0).argmax())); ys.append(organ_id[i])
        rows.append({"gsm": r.gsm, "organ": r.organ, "read_acc": float((p[-1] == organ_id[i]).mean()), "sample_pred": ps[-1]})
    y, p = np.concatenate(y), np.concatenate(p)
    labels = np.unique(y)
    return {"read_acc": float((y == p).mean()), "read_macro_f1": float(f1_score(y, p, labels=labels, average="macro")),
            "sample_acc": float(np.mean(np.asarray(ys) == np.asarray(ps))), "n_reads": int(len(y)), "n_samples": len(ys)}, rows


def cfdna_composition(model, store, organs, max_reads):
    model.eval(); out = {}
    for i, r in store.samples.iterrows():
        lo, _ = predict_sample(model, store, i, max_reads)
        frac = torch.bincount(lo.argmax(1), minlength=len(organs)).float().cpu().numpy(); frac /= frac.sum()
        top = np.argsort(-frac)[:5]
        out[r.gsm] = {organs[j]: float(frac[j]) for j in top}
    return out


def run(cfg: dict) -> dict:
    root = Path(cfg["root"]); dev = "cuda"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    sp = pd.read_parquet(cfg["splits"])
    ref = sp[sp.role == "reference"]
    organs = sorted(ref.organ.unique()); fines = sorted(ref.cell_type.unique())
    oi = {o: i for i, o in enumerate(organs)}; fi = {f: i for i, f in enumerate(fines)}
    fine_to_organ = torch.tensor([oi[ref[ref.cell_type == f].organ.iloc[0]] for f in fines], device=dev)
    have = {p.stem for p in Path(cfg["reads"]).glob("*.npz") if not p.stem.endswith(".tmp")}
    sp = sp[sp.gsm.isin(have)]
    out = Path(cfg["out"]) / cfg["arm"] / f"seed_{cfg['seed']}"; out.mkdir(parents=True, exist_ok=True)

    index = CpGIndex(root / "data/hg38_cpg_index.npz")
    table, positions, use_emb = load_table(cfg["arm"], root / cfg["embedding_h5"], index, dev, cfg["seed"])
    model = SetClassifier(table, positions, len(organs), len(fines), use_embedding=use_emb, d_model=cfg["d_model"],
                          n_layers=cfg["n_layers"]).to(dev)
    tr = ReadStore(Path(cfg["reads"]), sp[sp.split == "train"], dev)
    va = ReadStore(Path(cfg["reads"]), sp[sp.split == "val"], dev)
    tr_org = np.array([oi[o] for o in tr.samples.organ]); tr_fine = torch.tensor([fi[c] for c in tr.samples.cell_type], device=dev)
    va_org = np.array([oi[o] for o in va.samples.organ])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    gen = torch.Generator().manual_seed(cfg["seed"])
    organ_of_sample = torch.tensor(tr_org, device=dev)
    best, log = -1.0, []
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        model.train()
        idx = tr.balanced_indices(tr_org, cfg["batch"], gen)
        s = tr.sample_of_read(idx)
        r, st, m = tr.batch(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, lf = model(r, st, m)
        loss = F.cross_entropy(lo.float(), organ_of_sample[s]) + 0.5 * F.cross_entropy(lf.float(), tr_fine[s])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            met, _ = evaluate(model, va, va_org, None, cfg["eval_reads"], len(organs))
            log.append({"step": step, "loss": float(loss), **{f"val_{k}": v for k, v in met.items()}, "sec": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
            if met["read_macro_f1"] > best:
                best = met["read_macro_f1"]; torch.save({k: v for k, v in model.state_dict().items()}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True), strict=False)
    res = {"arm": cfg["arm"], "seed": cfg["seed"], "best_val_read_macro_f1": best, "train_log": log}
    del tr
    te = ReadStore(Path(cfg["reads"]), sp[sp.split == "test"], dev)
    te_org = np.array([oi[o] for o in te.samples.organ])
    res["test"], res["test_samples"] = evaluate(model, te, te_org, None, cfg["eval_reads"], len(organs))
    ex = sp[sp.split.isin(["external_cfdna", "external_wbc"])]
    if len(ex):
        xs = ReadStore(Path(cfg["reads"]), ex, dev)
        res["external_composition_top5"] = {r.gsm: {"role": r.role, **cfdna_composition(model, xs, organs, cfg["eval_reads"])[r.gsm]}
                                           for r in xs.samples.itertuples()}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    return res
