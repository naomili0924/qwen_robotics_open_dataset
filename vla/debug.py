"""Debug checks for the training pipeline, each a few minutes at most.

    python -m vla.debug data     --run runs/debug          # inspect samples: prompts, token counts, targets, a figure
    python -m vla.debug forward  --run runs/debug          # one forward/backward: loss, grad norms, memory, NaNs
    python -m vla.debug overfit  --run runs/debug --steps 150   # fit 8 scenarios; the loss must collapse
    python -m vla.debug compare  --run runs/debug --steps 150   # overfit with every head/denoiser combination
    python -m vla.debug pipeline --run runs/debug --steps 20    # fit / save / reload / add a head / predict

All Config flags apply after the stage name (e.g. `--head flow --denoiser dit --backbone-mode frozen`).
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod.scenario import CURRENT  # noqa: E402
from vla.config import Config, parse_args  # noqa: E402
from vla.data import INPUT_COLUMNS, Collator, NavDataset, columns_for, fit_action_scale, load_split, to_device  # noqa: E402
from vla.model import NavPolicy  # noqa: E402
from vla.tasks import aux_tasks  # noqa: E402


def project(path_xy, camera, index=-1):
    """Scenario-frame ground points -> pixels in one of the past images."""
    K = np.array(camera["K"]).reshape(3, 3)
    cam_from_scn = np.linalg.inv(np.array(camera["T_scenario_from_camera"][index]).reshape(4, 4))
    p = np.c_[path_xy, np.zeros(len(path_xy)), np.ones(len(path_xy))] @ cam_from_scn.T
    front = p[:, 2] > 0.3
    uv = p[front, :3] @ K.T
    return uv[:, :2] / uv[:, 2:]


def figure(rows, preds, out, cfg):
    """Per scenario: the frame strip, and the current image with the recorded (green) and predicted (red) path."""
    fig, axs = plt.subplots(len(rows), 2, figsize=(22, 5.5 * len(rows)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[2.2, 1]))
    for r, row in enumerate(rows):
        imgs = row["past_images"][-cfg.frames:]
        w, h = imgs[0].size
        strip = np.concatenate([np.asarray(im.resize((w // 4, h // 4))) for im in imgs], axis=1)
        axs[r, 0].imshow(strip)
        axs[r, 0].set_title(f"{row['scenario_id']}: {len(imgs)} frames, oldest left", fontsize=9)
        axs[r, 0].axis("off")
        ax = axs[r, 1]
        ax.imshow(np.asarray(imgs[-1]))
        gt = np.stack([row["ego"]["x"], row["ego"]["y"]], 1)[CURRENT + 1:]
        for xy, col, lab in ((gt, "#2ca02c", "recorded"), (preds[r], "#d62728", "predicted")):
            if xy is None:
                continue
            uv = project(xy, row["camera"])
            ax.plot(uv[:, 0], uv[:, 1], ".-", color=col, lw=2, ms=6, label=lab)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=60)
    print("wrote", out)


def stage_data(cfg, n=4):
    ds = load_split(cfg, cfg.train_split)
    if cfg.action_scale <= 0:
        cfg.action_scale = fit_action_scale(ds.select_columns(["ego"]), cfg)
    print(f"{len(ds)} scenarios, action scale {cfg.action_scale:.2f} m")
    from transformers import AutoProcessor
    collate = Collator(AutoProcessor.from_pretrained(cfg.backbone), cfg)
    nav = NavDataset(ds.select_columns(INPUT_COLUMNS), cfg)
    items = [nav[i] for i in range(n)]
    batch = collate(items)
    print("prompt:", items[0]["prompt"])
    print("tokens per sample:", batch["attention_mask"].sum(1).tolist(), "image patches:", tuple(batch["pixel_values"].shape))
    t = batch["target"]
    print(f"targets (normalised): shape {tuple(t.shape)}, min {t.min():.2f} max {t.max():.2f}, "
          f"final displacement {t[:, -1].norm(dim=1).mul(cfg.action_scale).tolist()} m")
    figure([ds[i] for i in range(n)], [None] * n, Path(cfg.run) / "data_samples.png", cfg)


def stage_forward(cfg):
    tasks = aux_tasks(cfg.tasks)
    ds = load_split(cfg, cfg.train_split, columns_for(cfg, tasks))
    if cfg.action_scale <= 0:
        cfg.action_scale = fit_action_scale(ds, cfg)
    model = NavPolicy(cfg)
    model.train()  # HF models load in eval mode, which also switches gradient checkpointing off
    print(model.describe())
    collate = Collator(model.processor, cfg)
    batch = to_device(collate([NavDataset(ds, cfg, tasks)[i] for i in range(cfg.batch_size)]), model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    parts = model.losses(batch)
    loss = sum(parts.values())
    loss.backward()
    print("losses:", {k: round(v.item(), 4) for k, v in parts.items()})
    torch.cuda.synchronize()
    head, backbone = model.trainable_parameters()
    gh = torch.nn.utils.clip_grad_norm_(head, float("inf"))
    gb = torch.nn.utils.clip_grad_norm_(backbone, float("inf")) if backbone else torch.tensor(0.0)
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"loss {loss.item():.4f}, grad norm head {gh:.3f} backbone {gb:.3f}, non-finite grads: {bad or 'none'}")
    print(f"forward+backward {time.time() - t0:.1f} s for batch {cfg.batch_size}, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    with torch.no_grad():
        pred = model.predict(batch)["trajectory"]
    print("prediction shape", tuple(pred.shape), "first:", pred[0, :3].cpu().numpy().round(2).tolist())


def stage_overfit(cfg, n=8, steps=150):
    rows = load_split(cfg, cfg.train_split)
    rows = rows.select(range(n))
    if cfg.action_scale <= 0:
        cfg.action_scale = fit_action_scale(rows.select_columns(["ego"]), cfg)
    model = NavPolicy(cfg)
    model.train()
    print(model.describe())
    tasks = aux_tasks(cfg.tasks)
    collate = Collator(model.processor, cfg)
    loader = DataLoader(NavDataset(rows.select_columns(columns_for(cfg, tasks)), cfg, tasks), batch_size=cfg.batch_size,
                        shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    head, backbone = model.trainable_parameters()
    groups = [{"params": head, "lr": cfg.lr_head}] + ([{"params": backbone, "lr": cfg.lr_backbone}] if backbone else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    batches = [to_device(b, model.device) for b in loader]  # cache: the point is the optimisation, not the loading
    losses, t0 = [], time.time()
    for step in range(steps):
        batch = batches[step % len(batches)]
        loss = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head + backbone, cfg.clip_grad)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step + 1) % 25 == 0:
            print(f"step {step + 1}: loss {np.mean(losses[-25:]):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    ade = []
    preds = {}
    for batch in batches:
        pred = model.predict(batch)["trajectory"].cpu().numpy()
        for i, idx in enumerate(batch["index"].tolist()):
            gt = batch["target"][i].cpu().numpy() * cfg.action_scale
            ade.append(np.linalg.norm(pred[i] - gt, axis=1).mean())
            preds[idx] = pred[i]
    print(f"after {steps} steps on {n} scenarios: loss {np.mean(losses[-10:]):.4f} (first 10: {np.mean(losses[:10]):.4f}), "
          f"ADE on the fitted scenarios {np.mean(ade):.2f} m")
    figure([rows[i] for i in range(min(3, n))], [preds.get(i) for i in range(min(3, n))], Path(cfg.run) / "overfit.png", cfg)
    return float(np.mean(losses[:10])), float(np.mean(losses[-10:])), float(np.mean(ade))


def stage_pipeline(cfg, steps=20):
    """Exercise the NavigationPipeline API end to end on a handful of scenarios."""
    from vla.pipeline import NavigationPipeline
    run = Path(cfg.run)
    pipe = NavigationPipeline(cfg, limit_train=16, limit_val=8, steps=steps, eval_every=steps, save_every=10 ** 9,
                              eval_batches=2, eval_scenarios=8, log_every=5, warmup=2, batch_size=2, grad_accum=1)
    print(pipe)
    pipe.fit(run=str(run / "pipe_a"))
    pipe.save_pretrained(run / "pipe_a" / "final")

    again = NavigationPipeline.from_pretrained(run / "pipe_a" / "final", workers=2)
    again.add_task("collision")
    again.freeze_backbone()
    print(again)
    again.fit(run=str(run / "pipe_b"), steps=steps, limit_train=16, limit_val=8, eval_every=steps, eval_batches=2,
              eval_scenarios=8, log_every=5, warmup=2, batch_size=2, grad_accum=1)
    print("evaluate:", again.evaluate(split=cfg.val_split, scenarios=8, batches=4))

    rows = load_split(cfg, cfg.val_split)
    row = rows[0]
    hist = np.stack([row["ego"]["x"], row["ego"]["y"]], 1)[:CURRENT + 1]
    out = again.predict(row["past_images"], hist, row["goal"], rate_hz=row["rate_hz"])
    print("predict() ->", {k: (np.asarray(v).shape if np.ndim(v) else round(float(v), 3)) for k, v in out.items()})
    print("trajectory[:3] (m):", np.round(out["trajectory"][:3], 2).tolist(), " recorded:",
          np.round(np.stack([row["ego"]["x"], row["ego"]["y"]], 1)[CURRENT + 1:CURRENT + 4], 2).tolist())


def stage_compare(cfg, steps):
    out = {}
    for head, denoiser, modes in (("regression", "mlp", 1), ("regression", "mlp", 3), ("flow", "mlp", 1), ("flow", "dit", 1)):
        cfg.head, cfg.denoiser, cfg.modes = head, denoiser, modes
        name = f"{head}" + (f"/{denoiser}" if head == "flow" else f"x{modes}")
        print(f"=== {name}", flush=True)
        out[name] = stage_overfit(cfg, steps=steps)
        torch.cuda.empty_cache()
    print("\nhead            first-loss  last-loss  ADE(m)")
    for k, (a, b, c) in out.items():
        print(f"{k:15s} {a:10.4f} {b:10.4f} {c:7.2f}")
    json.dump(out, open(Path(cfg.run) / "compare.json", "w"), indent=2)


def main():
    stage = sys.argv[1]
    argv = sys.argv[2:]
    steps = 150
    if "--steps" in argv:  # the stage's own step count, distinct from Config.steps
        k = argv.index("--steps")
        steps = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    cfg = parse_args(argv, Config(run="runs/debug", workers=2))
    Path(cfg.run).mkdir(parents=True, exist_ok=True)
    {"data": lambda: stage_data(cfg), "forward": lambda: stage_forward(cfg),
     "overfit": lambda: stage_overfit(cfg, steps=steps), "compare": lambda: stage_compare(cfg, steps),
     "pipeline": lambda: stage_pipeline(cfg, steps)}[stage]()


if __name__ == "__main__":
    main()
