"""Sample-level training / prediction for the CpG-set (processed-beta) pilot.

Same protocol as the fragment trainer: class-balanced batches of samples, a random subset of K observed CpGs per sample
per step, loss on the SAMPLE label only, model selection on validation (fixed seeded CpG subsets identical across arms),
best checkpoint kept; test is never seen here.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from cfdna_origin.data.beta import BetaData
from cfdna_origin.training.trainer import LocusTable, _balanced_schedule, fixed_subset


def _tensors(data: BetaData, table: LocusTable, sample_ids, cols_list, device):
    rows, betas, covs, seg = [], [], [], []
    for i, (sid, cols) in enumerate(zip(sample_ids, cols_list)):
        r, b, c = data.tokens(sid, cols)
        rows.append(r); betas.append(b); covs.append(c); seg.append(np.full(len(r), i, np.int64))
    rows = np.concatenate(rows)
    t = lambda a: torch.from_numpy(np.concatenate(a)).to(device)  # noqa: E731
    return table.gather(rows), t(betas), t(covs), t(seg)


@torch.no_grad()
def predict(model, data: BetaData, table: LocusTable, sample_ids, *, max_tokens, chunk, seed, device) -> np.ndarray:
    model.eval()
    out = []
    for sid in sample_ids:
        cols = fixed_subset(data.observed(sid), sid, max_tokens, seed)
        if not len(cols):
            raise ValueError(f"{sid}: no observed CpGs")

        def chunks():
            for s in range(0, len(cols), chunk):
                emb, b, c, _ = _tensors(data, table, [sid], [cols[s : s + chunk]], device)
                yield emb, b, c

        out.append(model.predict_streaming(chunks()).float().cpu().numpy())
    return np.stack(out)


def train(model, data: BetaData, table: LocusTable, *, train_ids, train_labels, val_ids, val_labels, n_classes,
          cfg: dict, seed: int, device: str, log=print) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    B, K = cfg["samples_per_step"], cfg["tokens_per_sample"]
    steps_per_epoch = math.ceil(cfg["bags_per_sample_per_epoch"] * len(train_ids) / B)
    total = steps_per_epoch * cfg["max_epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = max(1, int(cfg.get("warmup_frac", 0.03) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    y = torch.as_tensor(train_labels, device=device)
    best, best_epoch, best_state, stale, history = -math.inf, -1, None, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        order = _balanced_schedule(train_labels, steps_per_epoch * B, rng)
        losses = []
        for step in range(steps_per_epoch):
            picks = order[step * B : (step + 1) * B]
            ids = [train_ids[i] for i in picks]
            cols = [rng.choice(data.observed(s), min(K, len(data.observed(s))), replace=False) for s in ids]
            emb, b, c, seg = _tensors(data, table, ids, cols, device)
            logits, _ = model(emb, b, c, seg, len(ids))
            loss = F.cross_entropy(logits.float(), y[torch.as_tensor(picks, device=device)],
                                   label_smoothing=cfg.get("label_smoothing", 0.0))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
            opt.step(); sched.step()
            losses.append(float(loss))
        vl = predict(model, data, table, val_ids, max_tokens=cfg["val_max_tokens"], chunk=cfg["eval_chunk"],
                     seed=cfg["eval_seed"], device=device)
        vp = vl.argmax(1)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": float(F.cross_entropy(torch.from_numpy(vl), torch.as_tensor(val_labels))),
               "val_macro_f1": float(f1_score(val_labels, vp, labels=np.arange(n_classes), average="macro",
                                              zero_division=0)),
               "val_accuracy": float((vp == val_labels).mean()), "sec": time.time() - t0}
        history.append(rec)
        score = -rec["val_loss"] if cfg.get("monitor") == "val_loss" else rec[cfg.get("monitor", "val_macro_f1")]
        if score > best + 1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        log(f"epoch {epoch} loss {rec['train_loss']:.4f} val_f1 {rec['val_macro_f1']:.4f} val_loss {rec['val_loss']:.4f}")
        if stale >= cfg["patience"]:
            break
    return {"best_state": best_state, "history": history, "best_epoch": best_epoch,
            "best_val": {"monitor": cfg.get("monitor", "val_macro_f1"), "value": best}}
