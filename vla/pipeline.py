"""A pipeline-style front end: build, fit, predict, save, reload.

    from vla.pipeline import NavigationPipeline

    pipe = NavigationPipeline(head="flow", denoiser="dit", tasks="trajectory,occupancy")
    pipe.fit(steps=2000, run="runs/flow_dit")                  # trains on cfg.data
    out = pipe.predict(images, ego_history, goal, speed, rate_hz=2.0)
    out["trajectory"]                                           # (10, 2) metres, robot frame
    pipe.save_pretrained("runs/flow_dit/final")

    pipe = NavigationPipeline.from_pretrained("runs/flow_dit/final")
    pipe.fit_rl(algo="grpo", steps=300, run="runs/grpo")        # PPO / GRPO with the collision evaluator as reward
    pipe.add_task("collision")                                  # new head on the learnt embedding
    pipe.freeze_backbone()
    pipe.fit(steps=300, run="runs/transfer_collision")          # trains the new head only
    pipe.evaluate(split="validation", scenarios=200)
"""
import json
from pathlib import Path

import numpy as np
import torch

from hnod.scenario import CURRENT
from vla.config import Config
from vla.data import Collator, NavDataset, columns_for, kinematic_vector, load_split, prompt_for, to_device
from vla.model import NavPolicy
from vla.tasks import TASKS, aux_tasks


class NavigationPipeline:
    def __init__(self, cfg=None, device="cuda", **overrides):
        cfg = cfg or Config()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"unknown option {k!r}")
            setattr(cfg, k, v)
        self.cfg = cfg
        self.model = NavPolicy(cfg, device)
        self.collate = Collator(self.model.processor, cfg)

    # ------------------------------------------------------------------ persistence
    @classmethod
    def from_pretrained(cls, directory, device="cuda", **overrides):
        cfg = Config.load(Path(directory) / "config.json")
        cfg.init_from, cfg.resume = str(directory), ""
        return cls(cfg, device, **overrides)

    def save_pretrained(self, directory):
        self.model.save(directory)

    # ------------------------------------------------------------------ composition
    def add_task(self, name):
        self.model.add_task(TASKS[name])
        self.cfg = self.model.cfg

    def freeze_backbone(self):
        self.model.backbone.requires_grad_(False)
        self.cfg.freeze_backbone = True

    @property
    def tasks(self):
        return self.cfg.tasks.split(",")

    # ------------------------------------------------------------------ training / evaluation
    def fit(self, train_ds=None, val_ds=None, **options):
        """Train with vla.train's loop; `options` override Config fields (steps, run, lr_head, ...)."""
        from vla.train import train
        for k, v in options.items():
            setattr(self.cfg, k, v)
        train(self.cfg, self.model, train_ds, val_ds)
        return self

    def fit_rl(self, algo="grpo", **options):
        """PPO / GRPO fine-tuning with vla.rl; `options` override Config fields (steps, run, reward, group_size, ...)."""
        from vla.rl import train_rl
        self.cfg.algo = algo
        for k, v in options.items():
            setattr(self.cfg, k, v)
        train_rl(self.cfg, self.model)
        return self

    def evaluate(self, split="validation", scenarios=None, batches=None):
        from vla.train import evaluate, make_loader
        rows = load_split(self.cfg, split)
        ds = NavDataset(rows.select_columns(columns_for(self.cfg, aux_tasks(self.cfg.tasks))), self.cfg,
                        aux_tasks(self.cfg.tasks))
        loader = make_loader(ds, self.cfg, self.collate, shuffle=False)
        return evaluate(self.model, loader, rows, self.cfg, batches, scenarios)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_rows(self, rows):
        """Dataset rows (dicts with past_images, ego, goal, rate_hz) -> list of per-row output dicts."""
        self.model.eval()
        tasks = aux_tasks(self.cfg.tasks)
        items = []
        for i, row in enumerate(rows):
            imgs = row["past_images"][-self.cfg.frames:]
            items.append(dict(index=i, scenario_id=row.get("scenario_id", str(i)), images=[im.convert("RGB") for im in imgs],
                              prompt=prompt_for(row, self.cfg), kin=kinematic_vector(row),
                              target=np.zeros((self.cfg.horizon, self.cfg.action_dim), np.float32),
                              aux={t.name: np.zeros(t.out_dim, np.float32) for t in tasks}))
        batch = to_device(self.collate(items), self.model.device)
        out = self.model.predict(batch)
        results = []
        for i in range(len(rows)):
            r = {"trajectory": out["trajectory"][i].cpu().numpy()}
            for t in tasks:
                r[t.name] = t.decode(out[t.name][i].cpu().numpy())
            results.append(r)
        return results

    def predict(self, images, ego_history, goal, speed=None, rate_hz=2.0):
        """One scenario from raw inputs.

        images: list of PIL images, oldest first (the last one is the current frame).
        ego_history: (CURRENT + 1, 2) past positions in the current robot frame, oldest first, last = (0, 0).
        goal: (x, y) in the robot frame.  speed: m/s at the current frame (from the history if omitted).
        """
        hist = np.asarray(ego_history, dtype=np.float64)
        if speed is None:
            v = (hist[-1] - hist[-2]) * rate_hz
        else:
            v = np.array([speed, 0.0])
        n_future = self.cfg.horizon
        ego = {"x": list(hist[:, 0]) + [0.0] * n_future, "y": list(hist[:, 1]) + [0.0] * n_future,
               "vx": [0.0] * CURRENT + [float(v[0])] + [0.0] * n_future,
               "vy": [0.0] * CURRENT + [float(v[1])] + [0.0] * n_future}
        row = {"past_images": list(images), "ego": ego, "goal": list(map(float, goal)), "rate_hz": float(rate_hz)}
        return self.predict_rows([row])[0]

    def __repr__(self):
        return f"NavigationPipeline({self.model.describe()})"
