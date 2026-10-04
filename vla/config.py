"""Training configuration: one dataclass, settable from the command line, saved with every checkpoint."""
import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- data
    data: str = "Jinyan0924/qwen_robotics_open_dataset"  # Hugging Face repo id, or a local directory of parquet shards
    config_name: str = "coda_2hz"       # dataset config when `data` is a repo id
    train_split: str = "train"
    val_split: str = "validation"
    cache_dir: str = "/dev/shm/hnod/hfcache"
    max_pixels: int = 448 * 448         # per image, before the vision encoder (28 px per token: 448^2 -> 256 tokens)
    min_pixels: int = 128 * 128
    frames: int = 11                    # how many of the 11 past+current images to feed (the most recent ones)
    ego_history: bool = True            # put the robot's past positions in the prompt
    action_scale: float = 0.0           # metres per unit of normalised action; 0 = fit from the training data
    # --- model
    backbone: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    backbone_mode: str = "lora"         # frozen | lora | full
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    tune_vision: bool = False           # also adapt the vision encoder (LoRA/full modes)
    gradient_checkpointing: bool = True
    head: str = "regression"            # regression | flow
    denoiser: str = "mlp"               # mlp | dit  (flow head only)
    modes: int = 1                      # regression head: number of trajectory hypotheses (winner-takes-all)
    head_dim: int = 512
    head_layers: int = 4                # DiT blocks, or hidden layers of the MLPs
    head_heads: int = 8
    pool_queries: int = 4               # learned queries that read the backbone tokens (regression / mlp denoiser)
    flow_steps: int = 10                # Euler steps when sampling the flow head
    flow_samples: int = 1               # samples drawn per scenario; >1 returns the medoid
    time_sampling: str = "uniform"      # uniform | beta  (flow-matching timestep distribution)
    horizon: int = 10                   # future steps predicted
    action_dim: int = 2                 # x, y
    tasks: str = "trajectory"           # comma-separated: trajectory plus any of vla.tasks.TASKS (occupancy, collision, ...)
    kinematic_input: bool = True        # feed past positions / speed / goal to the heads as numbers, not only as prompt text
    init_from: str = ""                 # load weights (adapters + heads) from this checkpoint, start a fresh optimiser
    freeze_backbone: bool = False       # after loading: train heads only (transfer test of a learnt embedding)
    # --- reinforcement learning (vla.rl)
    algo: str = "grpo"                  # grpo | ppo
    group_size: int = 8                 # trajectories sampled per scenario
    rl_epochs: int = 1                  # optimisation passes over each sampled batch
    clip_eps: float = 0.2               # PPO/GRPO ratio clipping
    kl_beta: float = 0.04               # weight of the KL penalty against the reference policy
    bc_weight: float = 0.0              # add this much supervised (imitation) loss to the RL objective
    value_coef: float = 0.5             # PPO: weight of the value-head loss
    reward: str = "collision=1,goal=0.5,imitation=0.2,smooth=0.1"  # weighted sum of vla.rewards terms
    rl_lr_head: float = 1e-5            # RL learning rates: policy steps must stay small relative to the action noise
    rl_lr_backbone: float = 1e-5
    rl_std_init: float = 0.1            # regression policy: initial action std (normalised units)
    rl_flow_noise: float = 0.3          # flow policy: diffusion coefficient of the sampling SDE
    # --- optimisation
    batch_size: int = 4
    grad_accum: int = 4
    steps: int = 2000
    lr_head: float = 3e-4
    lr_backbone: float = 1e-4           # LoRA; use ~2e-5 for full fine-tuning
    weight_decay: float = 0.01
    warmup: int = 100
    clip_grad: float = 1.0
    seed: int = 0
    workers: int = 8
    # --- bookkeeping
    run: str = "runs/default"
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 16              # validation loss over this many batches
    eval_scenarios: int = 64            # validation scenarios scored with the collision evaluator
    save_every: int = 500
    resume: str = ""                    # checkpoint directory, or "auto": <run>/last locally, else from hub_repo
    hub_repo: str = ""                  # HF model repo that mirrors <run>/last at every checkpoint (survives the machine)
    limit_train: int = 0                # debugging: use only this many training scenarios
    limit_val: int = 0

    def save(self, path):
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def parse_args(argv=None, defaults=None):
    """Command line flags for every Config field (booleans as --flag / --no-flag)."""
    base = defaults or Config()
    ap = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(Config):
        name = "--" + f.name.replace("_", "-")
        default = getattr(base, f.name)
        if f.type is bool:
            ap.add_argument(name, dest=f.name, action=argparse.BooleanOptionalAction, default=default)
        else:
            ap.add_argument(name, dest=f.name, type=f.type, default=default)
    return Config(**vars(ap.parse_args(argv)))
