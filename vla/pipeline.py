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

    @classmethod
    def from_hub(cls, repo, checkpoint, revision=None, cache="/dev/shm/nav_policy_checkpoints", device="cuda",
                 **overrides):
        """Load a checkpoint folder of a Hugging Face model repo, e.g. ("user/nav_policy", "e1_all_sqrt/best").

        revision: a commit of the repo (older versions of an overwritten folder such as <run>/last).
        Needs HF_TOKEN for a private repo.
        """
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo, revision=revision, allow_patterns=[f"{checkpoint}/*"],
                                  local_dir=Path(cache) / repo.replace("/", "__") / (revision or "main"))
        return cls.from_pretrained(Path(local) / checkpoint, device, **overrides)

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

    @torch.no_grad()
    def predict_path(self, images, goal=None, instruction=None, past_xy=None, velocity=None):
        """Path for one moment, for policies trained on per-frame data (data_format="frames").

        images: PIL images, oldest first, cfg.past_dt_s seconds apart; the last is the current view (fewer
            than cfg.frames is fine: the oldest is repeated).
        goal: (x, y) in metres in the robot frame (robot at (0, 0), x forward, y left), and / or
        instruction: text such as "Walk to the glass door on the left."  At least one of the two.
        past_xy: the robot's past positions in the current robot frame, oldest first, cfg.past_dt_s apart,
            ending with (0, 0); optional (a robot that just started: all zeros).
        velocity: (vx, vy) m/s in the robot frame; optional (derived from past_xy).
        Returns (cfg.horizon, 2) waypoints in metres, cfg.spacing_m apart along the path.
        """
        from vla.data import frames_prompt
        from hnod.windows import POINT_TEMPLATES
        cfg = self.cfg
        assert cfg.data_format == "frames", "predict_path is for per-frame policies; use predict() otherwise"
        assert goal is not None or instruction, "give a goal, an instruction, or both"
        n = CURRENT + 1
        past = np.zeros((n, 2), np.float32) if past_xy is None else np.asarray(past_xy, np.float32).reshape(-1, 2)[-n:]
        if len(past) < n:  # clamp to the earliest known position, as at the start of a training episode
            past = np.concatenate([np.repeat(past[:1], n - len(past), 0), past])
        if velocity is None:
            velocity = (past[-1] - past[-2]) / cfg.past_dt_s
        velocity = np.asarray(velocity, np.float32)
        imgs = [im.convert("RGB") for im in images][-cfg.frames:]
        imgs = [imgs[0]] * (cfg.frames - len(imgs)) + imgs
        if instruction and goal is not None:
            prompt = f"{instruction.strip().rstrip('.')}. It is at ({goal[0]:.1f}, {goal[1]:.1f})."
        elif instruction:
            prompt = instruction.strip()
        else:
            prompt = POINT_TEMPLATES[0].format(x=goal[0], y=goal[1])
        s = dict(past_xy=past, velocity=velocity, prompt=prompt)
        g = np.zeros(2, np.float32) if goal is None else np.asarray(goal, np.float32)
        item = dict(index=0, scenario_id="0", images=imgs, prompt=frames_prompt(s, cfg),
                    kin=np.r_[past.ravel(), velocity, g].astype(np.float32),
                    target=np.zeros((cfg.horizon, cfg.action_dim), np.float32), aux={})
        self.model.eval()
        batch = to_device(self.collate([item]), self.model.device)
        return self.model.predict(batch)["trajectory"][0, :, :2].float().cpu().numpy()

    @torch.no_grad()
    def predict_motion(self, images, final_image, embodiment="robot"):
        """Final-frame policies (window_mode="final_frame"): past views + the view at the end of the horizon ->
        (cfg.horizon, 2) positions in metres, evenly spaced in time over cfg.horizon_s seconds.

        images: PIL images cfg.past_dt_s apart, oldest first, the last is the current view.
        final_image: the view cfg.horizon_s seconds from now.  embodiment: "human" or "robot".
        """
        from vla.data import final_frame_item
        from hnod.windows import embodiment_prompt
        assert self.cfg.window_mode == "final_frame", "this checkpoint is not a final-frame policy"
        prompt = embodiment_prompt("person_walking" if embodiment.startswith(("human", "person")) else "wheeled_robot")
        self.model.eval()
        batch = to_device(self.collate([final_frame_item(images, final_image, prompt, self.cfg)]), self.model.device)
        return self.model.predict(batch)["trajectory"][0, :, :2].float().cpu().numpy()

    def __repr__(self):
        return f"NavigationPipeline({self.model.describe()})"
