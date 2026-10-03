"""PPO / GRPO fine-tuning of a navigation policy with the collision evaluator as the reward.

    python -m vla.rl --algo grpo --init-from runs/sft/last --run runs/grpo --steps 300
    python -m vla.rl --algo ppo  --init-from runs/sft/last --run runs/ppo  --steps 300

One scenario is one decision (a whole trajectory), so this is the single-step case of
PPO/GRPO as used for language models:
  * sample `group_size` trajectories per scenario from the stochastic policy
    (Gaussian / mixture for the regression head, an SDE sampler for the flow head);
  * reward each with vla.rewards (collision-free, goal, imitation, smoothness ...);
  * GRPO: advantage = (r - mean_group) / std_group;  PPO: advantage = r - V(s) from a value head;
  * clipped-ratio policy loss, KL penalty against the reference policy (the starting
    weights: LoRA adapters disabled, frozen copy of the head), optional supervised term.
"""
import copy
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla import rewards as R  # noqa: E402
from vla.config import Config, parse_args  # noqa: E402
from vla.data import Collator, NavDataset, columns_for, fit_action_scale, load_split, to_device  # noqa: E402
from vla.heads import ValueHead  # noqa: E402
from vla.model import NavPolicy  # noqa: E402
from vla.tasks import aux_tasks  # noqa: E402
from vla.train import evaluate, lr_at, make_loader, seed_all  # noqa: E402

REWARD_COLUMNS = ("future_tracks", "future_lidar", "static_map", "ego", "goal", "rate_hz", "scenario_id")


def advantages(rewards, algo, values=None):
    """(G, B) rewards -> (G, B) advantages. GRPO: normalised within each scenario's group; PPO: against V(s)."""
    if algo == "grpo":
        return (rewards - rewards.mean(0, keepdim=True)) / (rewards.std(0, keepdim=True) + 1e-4)
    adv = rewards - values[None]
    return (adv - adv.mean()) / (adv.std() + 1e-4)


def policy_objective(new_logp, old_logp, ref_logp, adv, clip_eps, kl_beta):
    """Clipped surrogate + KL penalty (k3 estimator). Returns (loss, kl, clip fraction)."""
    ratio = torch.exp(new_logp - old_logp)
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
    delta = ref_logp - new_logp
    kl = (torch.exp(delta) - delta - 1).mean()
    return -surrogate + kl_beta * kl, kl, ((ratio - 1).abs() > clip_eps).float().mean()


class Reference:
    """Log-probs under the policy as it was when RL started (for the KL penalty).

    The head is copied.  A LoRA backbone gets a second, frozen adapter holding the starting
    weights (just disabling the adapter would give the pre-fine-tuning base model instead).
    """

    def __init__(self, model):
        self.model = model
        self.head = copy.deepcopy(model.head).eval().requires_grad_(False)
        self.lora = model.cfg.backbone_mode == "lora" and not model.cfg.freeze_backbone
        if self.lora:
            bb = model.backbone
            bb.add_adapter("reference", bb.peft_config["default"])
            for m in bb.modules():
                if hasattr(m, "lora_A") and "reference" in m.lora_A:
                    m.lora_A["reference"].weight.data.copy_(m.lora_A["default"].weight.data)
                    m.lora_B["reference"].weight.data.copy_(m.lora_B["default"].weight.data)
            self._activate("default")

    def _activate(self, name):
        bb = self.model.backbone
        bb.set_adapter(name)  # peft also flips requires_grad to the active adapter; keep "default" the only trainable one
        for n, p in bb.named_parameters():
            if ".lora_" in n:
                p.requires_grad_(".reference." not in n)

    @torch.no_grad()
    def log_prob(self, batch, sample):
        if self.lora:
            self._activate("reference")
        try:
            cond = self.model.encode(batch)
        finally:
            if self.lora:
                self._activate("default")
        return self.head.log_prob(cond, sample)


def train_rl(cfg: Config, model=None, train_rows=None, val_rows=None):
    seed_all(cfg.seed)
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    weights = R.parse_spec(cfg.reward)
    tasks = aux_tasks(cfg.tasks)
    if train_rows is None:
        train_rows = load_split(cfg, cfg.train_split)
    if val_rows is None:
        val_rows = load_split(cfg, cfg.val_split)
    cols = columns_for(cfg, tasks)
    if cfg.action_scale <= 0:
        cfg.action_scale = fit_action_scale(train_rows.select_columns(["ego"]), cfg)
    cfg.save(run / "config.json")
    reward_rows = train_rows.select_columns([c for c in REWARD_COLUMNS if c in train_rows.column_names])

    if model is None:
        model = NavPolicy(cfg)
    model.train()
    for m in model.modules():  # dropout would make the update's log-probs disagree with the rollout's
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    print(model.describe(), f"| {cfg.algo.upper()}, group {cfg.group_size}, reward {cfg.reward}", flush=True)
    collate = Collator(model.processor, cfg)
    train_loader = make_loader(NavDataset(train_rows.select_columns(cols), cfg, tasks), cfg, collate, shuffle=True,
                               limit=cfg.limit_train)
    val_loader = make_loader(NavDataset(val_rows.select_columns(cols), cfg, tasks), cfg, collate, shuffle=False,
                             limit=cfg.limit_val)
    reference = Reference(model)
    value = ValueHead(model.hidden, cfg).to(model.device) if cfg.algo == "ppo" else None

    head_params, backbone_params = model.trainable_parameters()
    if value is not None:
        head_params = head_params + list(value.parameters())
    groups = [{"params": head_params, "lr": cfg.rl_lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.rl_lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, cfg))
    writer = SummaryWriter(run / "tb")
    log = open(run / "log.jsonl", "a")
    step = 0

    def record(d):
        d["step"] = step
        for k, v in d.items():
            if isinstance(v, (int, float)) and k != "step":
                writer.add_scalar(k, v, step)
        log.write(json.dumps(d) + "\n")
        log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)

    def save(name):
        model.save(run / name)
        if value is not None:
            torch.save(value.state_dict(), run / name / "value_head.pt")

    stats, t0 = [], time.time()
    done = False
    while not done:
        for batch in train_loader:
            batch = to_device(batch, model.device)
            G, B = cfg.group_size, batch["target"].shape[0]

            # ---- rollout: sample, score, baseline
            model.eval()
            with torch.no_grad():
                cond = model.encode(batch)
                sample = model.head.sample(cond, G)
                old_logp = sample["logp"]                                   # (G, B)
                ref_logp = reference.log_prob(batch, sample)
                values = value(cond) if value is not None else None        # (B,)
            model.train()
            actions_m = (sample["actions"] * cfg.action_scale).cpu().numpy()
            rewards = np.zeros((G, B))
            parts = {k: [] for k in weights}
            for b, idx in enumerate(batch["index"].tolist()):
                row = reward_rows[idx]
                for g in range(G):
                    rewards[g, b], p = R.compute(row, actions_m[g, b, :, :2], weights)
                    for k, v in p.items():
                        parts[k].append(v)
            rewards_t = torch.tensor(rewards, device=model.device, dtype=torch.float32)
            adv = advantages(rewards_t, cfg.algo, values)

            # ---- update
            for _ in range(cfg.rl_epochs):
                cond = model.encode(batch)
                new_logp = model.head.log_prob(cond, sample)
                loss, kl, clip_frac = policy_objective(new_logp, old_logp, ref_logp, adv, cfg.clip_eps, cfg.kl_beta)
                vloss = torch.tensor(0.0)
                if value is not None:
                    vloss = ((value(cond)[None] - rewards_t) ** 2).mean()
                    loss = loss + cfg.value_coef * vloss
                bc = torch.tensor(0.0)
                if cfg.bc_weight > 0:
                    bc = model.head.loss(cond, batch["target"].float())
                    loss = loss + cfg.bc_weight * bc
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(head_params + backbone_params, cfg.clip_grad)
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            stats.append(dict(reward=rewards.mean(), reward_std_in_group=rewards.std(0).mean(), kl=kl.item(),
                              clip_frac=clip_frac.item(), grad_norm=float(gnorm),
                              value_loss=vloss.item(), bc_loss=bc.item(),
                              **{f"reward_{k}": float(np.mean(v)) for k, v in parts.items()}))
            if step % cfg.log_every == 0:
                d = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
                d.update(sec_per_step=(time.time() - t0) / cfg.log_every, gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                         lr_head=sched.get_last_lr()[0])
                if cfg.head == "regression":
                    d["action_std"] = model.head.log_std.exp().mean().item()
                record(d)
                stats, t0 = [], time.time()
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
    train_rl(parse_args())
