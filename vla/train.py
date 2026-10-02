"""Train a navigation policy: python -m vla.train --run runs/x --head flow --denoiser dit ...

Logs to <run>/log.jsonl and TensorBoard (<run>/tb), checkpoints to <run>/step_N and <run>/last.
The same loop is used by NavigationPipeline.fit().
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import eval as ev  # noqa: E402
from vla.config import Config, parse_args  # noqa: E402
from vla.data import Collator, NavDataset, columns_for, fit_action_scale, load_split, to_device  # noqa: E402
from vla.model import NavPolicy  # noqa: E402
from vla.tasks import aux_tasks  # noqa: E402


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(ds, cfg, collate, shuffle, limit=0):
    if limit:
        ds = Subset(ds, list(range(min(limit, len(ds)))))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate, num_workers=cfg.workers,
                      drop_last=shuffle, persistent_workers=cfg.workers > 0)


def lr_at(step, cfg):
    if step < cfg.warmup:
        return step / max(1, cfg.warmup)
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, loader, val_rows, cfg, max_batches=None, max_scenarios=None):
    """Validation losses, trajectory ADE/FDE and collision rates (via hnod.eval), auxiliary task metrics."""
    model.eval()
    tasks = {t.name: t for t in aux_tasks(cfg.tasks)}
    losses, results, aux_pred, aux_true, seen = [], [], {t: [] for t in tasks}, {t: [] for t in tasks}, 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        batch = to_device(batch, model.device)
        losses.append({k: v.item() for k, v in model.losses(batch).items()})
        if max_scenarios is not None and seen >= max_scenarios:
            continue
        pred = model.predict(batch)
        traj = pred["trajectory"].cpu().numpy()
        for i, idx in enumerate(batch["index"].tolist()):
            if max_scenarios is not None and seen >= max_scenarios:
                break
            if "trajectory" in cfg.tasks:
                results.append(ev.evaluate_scenario(val_rows[idx], traj[i, :, :2]))
            for t in tasks:
                aux_pred[t].append(pred[t][i].cpu().numpy())
                aux_true[t].append(batch["aux"][t][i].cpu().numpy())
            seen += 1
    model.train()
    out = {}
    if losses:
        out["val_loss"] = float(np.mean([sum(l.values()) for l in losses]))
        for k in losses[0]:
            out[f"val_loss_{k}"] = float(np.mean([l[k] for l in losses]))
    if results:
        out.update({k: v for k, v in ev.aggregate(results).items() if k != "num_scenarios"})
    for t, task in tasks.items():
        if aux_pred[t]:
            out.update(task.metrics(np.stack(aux_pred[t]), np.stack(aux_true[t])))
    out["eval_scenarios"] = seen
    return out


def train(cfg: Config, model=None, train_ds=None, val_rows=None):
    """Training loop.  train_ds / val_rows default to cfg.data's splits (val_rows keeps every column)."""
    seed_all(cfg.seed)
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    tasks = aux_tasks(cfg.tasks)
    cols = columns_for(cfg, tasks)
    if train_ds is None:
        train_ds = load_split(cfg, cfg.train_split, cols)
    if val_rows is None:
        val_rows = load_split(cfg, cfg.val_split)  # all columns: the evaluator needs the ground truth
    val_ds = val_rows.select_columns(cols)
    if cfg.action_scale <= 0:
        cfg.action_scale = fit_action_scale(train_ds, cfg)
    cfg.save(run / "config.json")
    print(f"train {len(train_ds)} val {len(val_ds)} scenarios, action scale {cfg.action_scale:.2f} m", flush=True)

    if model is None:
        model = NavPolicy(cfg)
    print(model.describe(), flush=True)
    collate = Collator(model.processor, cfg)
    train_loader = make_loader(NavDataset(train_ds, cfg, tasks), cfg, collate, shuffle=True, limit=cfg.limit_train)
    val_loader = make_loader(NavDataset(val_ds, cfg, tasks), cfg, collate, shuffle=False, limit=cfg.limit_val)

    head_params, backbone_params = model.trainable_parameters()
    groups = [{"params": head_params, "lr": cfg.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, cfg))
    step = 0
    if cfg.resume:
        model.load(cfg.resume)
        state = torch.load(Path(cfg.resume) / "trainer.pt", map_location="cpu")
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        print(f"resumed from {cfg.resume} at step {step}", flush=True)

    writer = SummaryWriter(run / "tb")
    log = open(run / "log.jsonl", "a")

    def record(d):
        d["step"] = step
        for k, v in d.items():
            if isinstance(v, (int, float)) and k != "step":
                writer.add_scalar(k, v, step)
        log.write(json.dumps(d) + "\n")
        log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)

    def save(name):
        d = run / name
        model.save(d)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, d / "trainer.pt")

    model.train()
    t0, window = time.time(), []
    done = False
    while not done:
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, model.device)
            parts = model.losses(batch)
            loss = sum(parts.values()) / cfg.grad_accum
            loss.backward()
            window.append({k: v.item() for k, v in parts.items()})
            if (i + 1) % cfg.grad_accum:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(head_params + backbone_params, cfg.clip_grad)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0:
                d = {"loss": float(np.mean([sum(w.values()) for w in window])), "grad_norm": float(gnorm),
                     "lr_head": sched.get_last_lr()[0], "sec_per_step": (time.time() - t0) / cfg.log_every,
                     "gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
                if len(window[0]) > 1:
                    d.update({f"loss_{k}": float(np.mean([w[k] for w in window])) for k in window[0]})
                record(d)
                window, t0 = [], time.time()
            if step % cfg.eval_every == 0:
                record(evaluate(model, val_loader, val_rows, cfg, cfg.eval_batches, cfg.eval_scenarios))
                t0 = time.time()
            if step % cfg.save_every == 0:
                save(f"step_{step}")
            if step >= cfg.steps:
                done = True
                break
    save("last")
    record(evaluate(model, val_loader, val_rows, cfg, None, cfg.eval_scenarios))
    print(f"done: {run}", flush=True)
    return model


if __name__ == "__main__":
    train(parse_args())
