"""Sample-level MIL training and deterministic prediction.

Separation of concerns (CPU vs GPU):
  FragmentStore (CPU, mmap)  -> bag of fragment indices -> padded arrays
  LocusTable (CPU memmap, or GPU if small) -> gather the embedding rows the bag needs
  MILClassifier (GPU)        -> fragment encoder + sample aggregator

Training: each step draws B samples (class-balanced) and K random valid fragments per sample; the loss is the
cross-entropy of the SAMPLE label. Model selection uses validation only (fixed seeded fragment subsets, identical for
every arm); the best checkpoint is kept. The trainer never receives test samples or labels.
"""
from __future__ import annotations

import math
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from cfdna_origin.data.fragments import FragmentStore


class LocusTable:
    """Frozen materialised locus table [n_loci, r]. On GPU when it fits `gpu_max_gb`, else gathered on CPU."""

    def __init__(self, table: np.ndarray, device: str, gpu_max_gb: float = 4.0):
        self.dim = int(table.shape[1])
        self.device = device
        nbytes = table.size * 2
        self.on_device = self.dim > 0 and device != "cpu" and nbytes <= gpu_max_gb * 1e9
        if self.on_device:
            self.table = torch.from_numpy(np.ascontiguousarray(table)).to(device)
        else:
            self.table = table  # numpy (memmap) on CPU

    def gather(self, rows: np.ndarray) -> torch.Tensor | None:
        if self.dim == 0:
            return None
        if self.on_device:
            return self.table[torch.from_numpy(rows).to(self.device)]
        uniq, inv = np.unique(rows, return_inverse=True)
        vals = torch.from_numpy(np.asarray(self.table[uniq]))
        return vals[torch.from_numpy(inv.reshape(rows.shape))].to(self.device)


def _to_device(bag, table: LocusTable, device: str):
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return table.gather(bag.rows), t(bag.state), t(bag.geom), t(bag.mask), t(bag.sample)


def fixed_subset(valid: np.ndarray, sample_id: str, max_fragments: int | None, seed: int) -> np.ndarray:
    """Deterministic fragment subset of a sample: depends only on (sample_id, seed), identical across arms."""
    if not max_fragments or len(valid) <= max_fragments:
        return valid
    rng = np.random.default_rng([seed, zlib.crc32(sample_id.encode())])
    return np.sort(rng.choice(valid, max_fragments, replace=False))


@torch.no_grad()
def predict(model, store: FragmentStore, table: LocusTable, sample_ids: list[str], *, max_fragments: int | None,
            chunk: int, seed: int, device: str, heldout=None, heldout_mode: str = "exclude") -> np.ndarray:
    """Exact streaming prediction; returns logits [n_samples, C]."""
    model.eval()
    out = []
    for sid in sample_ids:
        valid = store.valid_fragments(sid, heldout, heldout_mode) if heldout is not None else store.valid_fragments(sid)
        idx = fixed_subset(valid, sid, max_fragments, seed)
        if not len(idx):
            raise ValueError(f"{sid}: no valid fragments to evaluate")

        def chunks():
            for s in range(0, len(idx), chunk):
                emb, st, ge, ma, _ = _to_device(store.bag([sid], [idx[s : s + chunk]]), table, device)
                yield emb, st, ge, ma

        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out.append(model.predict_streaming(chunks()).float().cpu().numpy())
    return np.stack(out)


def _balanced_schedule(labels: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    w = (1.0 / counts)[np.searchsorted(classes, labels)]
    return rng.choice(len(labels), n_draws, replace=True, p=w / w.sum())


def train(model, store: FragmentStore, table: LocusTable, *, train_ids: list[str], train_labels: np.ndarray,
          val_ids: list[str], val_labels: np.ndarray, n_classes: int, cfg: dict, seed: int, device: str,
          heldout=None, log=print) -> dict:
    """Returns {"best_state", "history", "best_epoch", "best_val"}; the caller reloads best_state."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    B, K = cfg["samples_per_step"], cfg["fragments_per_bag"]
    draws_per_epoch = cfg["bags_per_sample_per_epoch"] * len(train_ids)
    steps_per_epoch = math.ceil(draws_per_epoch / B)
    total = steps_per_epoch * cfg["max_epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = max(1, int(cfg.get("warmup_frac", 0.03) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    valid = {s: (store.valid_fragments(s, heldout, "exclude") if heldout is not None else store.valid_fragments(s))
             for s in train_ids}
    empty = [s for s, v in valid.items() if not len(v)]
    if empty:
        raise ValueError(f"training samples without valid fragments: {empty[:5]}")
    y_train = torch.as_tensor(train_labels, device=device)
    monitor = cfg.get("monitor", "val_macro_f1")
    best, best_epoch, best_state, stale, history = -math.inf, -1, None, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        order = _balanced_schedule(train_labels, steps_per_epoch * B, rng) if cfg.get("class_balanced", True) \
            else rng.permutation(np.resize(np.arange(len(train_ids)), steps_per_epoch * B))
        losses = []
        for step in range(steps_per_epoch):
            picks = order[step * B : (step + 1) * B]
            ids = [train_ids[i] for i in picks]
            frag = [rng.choice(valid[s], K, replace=len(valid[s]) < K) for s in ids]
            emb, st, ge, ma, seg = _to_device(store.bag(ids, frag), table, device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, _ = model(emb, st, ge, ma, seg, len(ids))
            loss = F.cross_entropy(logits.float(), y_train[torch.as_tensor(picks, device=device)],
                                   label_smoothing=cfg.get("label_smoothing", 0.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
            opt.step(); sched.step()
            losses.append(float(loss))
        val_logits = predict(model, store, table, val_ids, max_fragments=cfg["val_max_fragments"],
                             chunk=cfg["eval_chunk"], seed=cfg["eval_seed"], device=device,
                             heldout=heldout, heldout_mode="exclude")  # held-out loci stay unseen during selection
        val_pred = val_logits.argmax(1)
        val_loss = float(F.cross_entropy(torch.from_numpy(val_logits), torch.as_tensor(val_labels)))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_macro_f1": float(f1_score(val_labels, val_pred, labels=np.arange(n_classes), average="macro",
                                              zero_division=0)),
               "val_accuracy": float((val_pred == val_labels).mean()), "lr": sched.get_last_lr()[0],
               "sec": time.time() - t0}
        history.append(rec)
        score = -rec["val_loss"] if monitor == "val_loss" else rec[monitor]
        improved = score > best + 1e-6
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        log(f"epoch {epoch} loss {rec['train_loss']:.4f} val_loss {val_loss:.4f} val_f1 {rec['val_macro_f1']:.4f}"
            f"{' *' if improved else ''}")
        if stale >= cfg["patience"]:
            break
    return {"best_state": best_state, "history": history, "best_epoch": best_epoch,
            "best_val": {"monitor": monitor, "value": best}}
