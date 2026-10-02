"""Representation probing: is the frozen backbone's embedding enough for a simple head?

    python -m vla.probe --data /path/to/coda_2hz --run runs/probe --train-samples 600 --val-samples 200

For every layer of the frozen backbone, three pooled features are extracted
(last prompt token, mean over image tokens, mean over text tokens) and cheap
probes are fitted on them: a ridge (linear) probe and a two-layer MLP probe.
Targets:
  trajectory  - the 10 future positions (what the policy must output)
  residual    - trajectory minus the straight line to the goal: the part that needs
                the scene (obstacle avoidance), not just kinematics and the goal
  occupancy   - fraction of occupied map cells in an 8 x 8 grid of the 8 m ahead:
                does the embedding encode where the obstacles are?
Reference points: the same probes on kinematic inputs only (past positions, speed,
goal), a linear probe on kinematics plus the layer's features ("linear+kin": does
vision add anything on top of the numbers?), and the straight-to-goal rule.  Reading the result:
  * linear probe close to MLP probe        -> the feature is linearly usable; an MLP head is enough
  * MLP probe far better than linear       -> information is there but entangled; a deeper head helps
  * neither beats the kinematic baseline   -> the frozen embedding lacks the information;
                                              adapt the backbone (LoRA/full) or give the head
                                              token-level access (DiT cross-attention)
  * best layer well before the last        -> late layers specialise for language; tap earlier
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod.eval import decode_map  # noqa: E402
from hnod.maps import MAP_RES, MAP_SIZE, PIX_OCCUPIED  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402
from vla.config import Config, parse_args  # noqa: E402
from vla.data import INPUT_COLUMNS, Collator, NavDataset, fit_action_scale, future_xy, load_split, to_device  # noqa: E402
from vla.model import INPUT_KEYS, NavPolicy  # noqa: E402

POOLS = ("last", "image", "text")
OCC_CELLS, OCC_RANGE = 8, 8.0  # 8 x 8 cells over x in [0, 8], y in [-4, 4]


# ----------------------------------------------------------------------------- targets
def occupancy_ahead(row):
    img = decode_map(row["static_map"])
    occ = (img == PIX_OCCUPIED).astype(np.float32)
    r0, r1 = int(MAP_SIZE / 2 - OCC_RANGE / MAP_RES), int(MAP_SIZE / 2)          # rows: x in [0, 8] is above the centre
    c0, c1 = int(MAP_SIZE / 2 - OCC_RANGE / 2 / MAP_RES), int(MAP_SIZE / 2 + OCC_RANGE / 2 / MAP_RES)
    block = occ[r0:r1, c0:c1]
    k = block.shape[0] // OCC_CELLS
    return block.reshape(OCC_CELLS, k, OCC_CELLS, k).mean((1, 3)).ravel()


def kinematic_inputs(row):
    ego = row["ego"]
    past = np.stack([ego["x"], ego["y"]], 1)[:CURRENT + 1].ravel()
    return np.r_[past, ego["vx"][CURRENT], ego["vy"][CURRENT], row["goal"]].astype(np.float32)


def straight_to_goal(row, cfg):
    return np.arange(1, cfg.horizon + 1)[:, None] / cfg.horizon * np.asarray(row["goal"])[None]


def targets_for(rows, cfg):
    traj = np.stack([future_xy(r, cfg) for r in rows]).astype(np.float32)
    straight = np.stack([straight_to_goal(r, cfg) for r in rows]).astype(np.float32)
    return {"trajectory": traj.reshape(len(rows), -1) / cfg.action_scale,
            "residual": (traj - straight).reshape(len(rows), -1) / cfg.action_scale,
            "occupancy": np.stack([occupancy_ahead(r) for r in rows])}


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def extract(model, ds, cfg, indices):
    """(N, layers, pools, H) float16 pooled hidden states of the frozen backbone."""
    collate = Collator(model.processor, cfg)
    loader = DataLoader(Subset(NavDataset(ds, cfg), indices), batch_size=cfg.batch_size, collate_fn=collate,
                        num_workers=cfg.workers)
    image_token = model.inner.config.image_token_id
    out = []
    for batch in loader:
        batch = to_device(batch, model.device)
        inputs = {k: batch[k] for k in INPUT_KEYS}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model.backbone(**inputs, output_hidden_states=True, use_cache=False).hidden_states
        mask = batch["attention_mask"].bool()
        img = (batch["input_ids"] == image_token) & mask
        txt = mask & ~img
        last = mask.sum(1) - 1
        feats = []
        for h in hs:
            h = h.float()
            feats.append(torch.stack([h[torch.arange(len(h)), last],
                                      (h * img[..., None]).sum(1) / img.sum(1, keepdim=True),
                                      (h * txt[..., None]).sum(1) / txt.sum(1, keepdim=True)], 1))
        out.append(torch.stack(feats, 1).half().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- probes
def standardise(xtr, xva):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    return (xtr - mu) / sd, (xva - mu) / sd


def ridge(xtr, ytr, xva, lambdas=(1e-2, 1e-1, 1, 10, 100, 1000)):
    """Closed-form ridge regression; lambda picked on a fifth of the training rows."""
    xtr, xva = standardise(xtr, xva)
    n = len(xtr)
    cut = n - n // 5
    best = None
    xt, yt = torch.tensor(xtr, device="cuda", dtype=torch.float64), torch.tensor(ytr, device="cuda", dtype=torch.float64)
    xv = torch.tensor(xva, device="cuda", dtype=torch.float64)
    ymu = yt[:cut].mean(0)
    for lam in lambdas:
        a, b = xt[:cut], yt[:cut] - ymu
        w = torch.linalg.solve(a.T @ a + lam * torch.eye(a.shape[1], device=a.device, dtype=a.dtype), a.T @ b)
        err = ((xt[cut:] @ w + ymu - yt[cut:]) ** 2).mean().item()
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    ymu = yt.mean(0)
    w = torch.linalg.solve(xt.T @ xt + lam * torch.eye(xt.shape[1], device=xt.device, dtype=xt.dtype), xt.T @ (yt - ymu))
    return (xv @ w + ymu).float().cpu().numpy()


def mlp_probe(xtr, ytr, xva, hidden=512, epochs=300, lr=1e-3, wd=1e-2, seed=0):
    xtr, xva = standardise(xtr, xva)
    torch.manual_seed(seed)
    xt, yt = torch.tensor(xtr, device="cuda", dtype=torch.float32), torch.tensor(ytr, device="cuda", dtype=torch.float32)
    net = nn.Sequential(nn.Linear(xt.shape[1], hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, yt.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(net(xt), yt).backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(xva, device="cuda", dtype=torch.float32)).cpu().numpy()


def score(pred, true, target, cfg):
    """ADE in metres for trajectory targets, R^2 for occupancy."""
    if target == "occupancy":
        ss = ((true - pred) ** 2).sum()
        return float(1 - ss / ((true - true.mean(0)) ** 2).sum())
    d = (pred - true).reshape(len(true), -1, cfg.action_dim) * cfg.action_scale
    return float(np.linalg.norm(d, axis=2).mean())


# ----------------------------------------------------------------------------- main
def main():
    argv = sys.argv[1:]
    extra = argparse.ArgumentParser()
    extra.add_argument("--train-samples", type=int, default=600)
    extra.add_argument("--val-samples", type=int, default=200)
    extra.add_argument("--layer-step", type=int, default=1, help="probe every k-th layer")
    ex, rest = extra.parse_known_args(argv)
    cfg = parse_args(rest, Config(run="runs/probe", backbone_mode="frozen", gradient_checkpointing=False, batch_size=8,
                                  workers=4))
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)

    rows = {s: load_split(cfg, getattr(cfg, f"{s}_split")) for s in ("train", "val")}
    if cfg.action_scale <= 0:
        cfg.action_scale = fit_action_scale(rows["train"].select_columns(["ego"]), cfg)
    rng = np.random.default_rng(cfg.seed)
    idx = {"train": np.sort(rng.choice(len(rows["train"]), min(ex.train_samples, len(rows["train"])), replace=False)),
           "val": np.sort(rng.choice(len(rows["val"]), min(ex.val_samples, len(rows["val"])), replace=False))}

    feats, targets, kin = {}, {}, {}
    cache = run / "features.npz"
    cached = np.load(cache) if cache.exists() else None
    if cached is not None and np.array_equal(cached["train_idx"], idx["train"]) and np.array_equal(cached["val_idx"], idx["val"]):
        feats = {"train": cached["train"], "val": cached["val"]}
        print("reusing", cache, flush=True)
    else:
        model = NavPolicy(cfg)
        model.eval()
        for s in ("train", "val"):
            t0 = time.time()
            feats[s] = extract(model, rows[s].select_columns(INPUT_COLUMNS), cfg, idx[s].tolist())
            print(f"{s}: features {feats[s].shape} in {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(cache, train=feats["train"], val=feats["val"], train_idx=idx["train"], val_idx=idx["val"])
    for s in ("train", "val"):
        sub = [rows[s][int(i)] for i in idx[s]]
        targets[s] = targets_for(sub, cfg)
        kin[s] = np.stack([kinematic_inputs(r) for r in sub])
    n_layers = feats["train"].shape[1]
    layers = list(range(0, n_layers, ex.layer_step))
    if layers[-1] != n_layers - 1:
        layers.append(n_layers - 1)

    results = {"config": {"backbone": cfg.backbone, "layers": layers, "pools": POOLS, "train": len(idx["train"]),
                          "val": len(idx["val"]), "action_scale": cfg.action_scale}, "baselines": {}, "layers": {}}
    # Reference points without any vision
    straight = np.stack([straight_to_goal(rows["val"][int(i)], cfg) for i in idx["val"]]).reshape(len(idx["val"]), -1) / cfg.action_scale
    for target in ("trajectory", "residual", "occupancy"):
        tr, va = targets["train"][target], targets["val"][target]
        results["baselines"][target] = {
            "kinematic_linear": score(ridge(kin["train"], tr, kin["val"]), va, target, cfg),
            "kinematic_mlp": score(mlp_probe(kin["train"], tr, kin["val"]), va, target, cfg),
            "predict_mean": score(np.tile(tr.mean(0), (len(va), 1)), va, target, cfg)}
        if target == "trajectory":
            results["baselines"][target]["straight_to_goal"] = score(straight, va, target, cfg)
        if target == "residual":
            results["baselines"][target]["straight_to_goal"] = score(np.zeros_like(va), va, target, cfg)
    print("baselines:", json.dumps(results["baselines"], indent=1), flush=True)

    t0 = time.time()
    for layer in layers:
        res = {}
        for p, pool in enumerate(POOLS):
            xtr, xva = feats["train"][:, layer, p].astype(np.float32), feats["val"][:, layer, p].astype(np.float32)
            for target in ("trajectory", "residual", "occupancy"):
                tr, va = targets["train"][target], targets["val"][target]
                res[f"{pool}/{target}/linear"] = score(ridge(xtr, tr, xva), va, target, cfg)
                res[f"{pool}/{target}/mlp"] = score(mlp_probe(xtr, tr, xva), va, target, cfg)
                # features next to the kinematic numbers: does vision add anything on top of them?
                res[f"{pool}/{target}/linear+kin"] = score(
                    ridge(np.c_[kin["train"], xtr], tr, np.c_[kin["val"], xva]), va, target, cfg)
        results["layers"][layer] = res
        print(f"layer {layer:2d}: " + "  ".join(f"{k}={v:.2f}" for k, v in res.items() if k.startswith("last/")), flush=True)
    print(f"probes fitted in {time.time() - t0:.0f} s")
    json.dump(results, open(run / "probe.json", "w"), indent=1)
    report(results, run, cfg)


def report(results, run, cfg):
    layers = results["config"]["layers"]
    base = results["baselines"]
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5))
    for ax, target in zip(axs, ("trajectory", "residual", "occupancy")):
        for pool, ls in (("last", "-"), ("image", "--"), ("text", ":")):
            for probe, col in (("linear", "#1f77b4"), ("mlp", "#d62728"), ("linear+kin", "#9467bd")):
                ax.plot(layers, [results["layers"][l][f"{pool}/{target}/{probe}"] for l in layers], ls, color=col,
                        label=f"{pool} / {probe}")
        for name, col in (("kinematic_mlp", "k"), ("straight_to_goal", "g"), ("predict_mean", "0.5")):
            if name in base[target]:
                ax.axhline(base[target][name], color=col, lw=1, ls="-.", label=name)
        ax.set_title(f"{target}: " + ("R^2 (higher is better)" if target == "occupancy" else "ADE [m] (lower is better)"))
        ax.set_xlabel("backbone layer")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run / "probe.png", dpi=70)

    lines = [f"backbone {results['config']['backbone']}, frozen; {results['config']['train']} train / "
             f"{results['config']['val']} val scenarios\n"]
    for target in ("trajectory", "residual", "occupancy"):
        better = max if target == "occupancy" else min
        best = {}
        for probe in ("linear", "mlp", "linear+kin"):
            cands = [(results["layers"][l][f"{pool}/{target}/{probe}"], l, pool) for l in layers for pool in results["config"]["pools"]]
            best[probe] = better(cands)
        unit = "R^2" if target == "occupancy" else "ADE m"
        lines.append(f"{target} ({unit}):")
        for probe in ("linear", "mlp", "linear+kin"):
            v, l, pool = best[probe]
            lines.append(f"  best {probe:10s} probe: {v:.3f}  (layer {l}, {pool} pooling)")
        for k, v in base[target].items():
            lines.append(f"  {k:18s}: {v:.3f}")
    txt = "\n".join(lines)
    (run / "probe_summary.txt").write_text(txt)
    print("\n" + txt)
    print("wrote", run / "probe.png")


if __name__ == "__main__":
    main()
