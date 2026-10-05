"""Scenario rows -> prompts, images and normalised action targets."""
import glob
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from hnod.scenario import CURRENT

INPUT_COLUMNS = ["scenario_id", "rate_hz", "past_images", "ego", "goal", "task", "instruction"]
OPTIONAL_COLUMNS = {"task", "instruction"}

SYSTEM_PROMPT = ("You are the navigation policy of a mobile robot. The images are the robot's front camera, "
                 "oldest first; the last one is the current moment. Coordinates are in metres in the robot's "
                 "current frame: x forward, y left.")


def load_split(cfg, split, columns=None):
    """A map-style datasets.Dataset for one split, local parquet directory or Hub repo."""
    if os.path.isdir(cfg.data):
        files = sorted(glob.glob(os.path.join(cfg.data, f"{split}-*.parquet")))
        ds = load_dataset("parquet", data_files=files, split="train", cache_dir=cfg.cache_dir)
    else:
        ds = load_dataset(cfg.data, cfg.config_name, split=split, cache_dir=cfg.cache_dir)
    if columns:  # datasets published before the task / instruction columns existed do not have them
        ds = ds.select_columns([c for c in columns if c in ds.column_names or c not in OPTIONAL_COLUMNS])
    return ds


def language_task(row):
    """True if the row's goal is given in words (VLN, object-goal) rather than as a point."""
    return bool(row.get("instruction")) and row.get("task", "pointgoal") != "pointgoal"


def prompt_for(row, cfg):
    ego = row["ego"]
    rate = row["rate_hz"]
    parts = [f"Frame interval {1.0 / rate:.1f} s. Current speed {np.hypot(ego['vx'][CURRENT], ego['vy'][CURRENT]):.2f} m/s."]
    if cfg.ego_history:
        past = ", ".join(f"({ego['x'][i]:.1f}, {ego['y'][i]:.1f})" for i in range(CURRENT + 1))
        parts.append(f"Robot positions over the last {CURRENT} frames, oldest first: {past}.")
    if language_task(row):
        parts.append(f"Instruction: {row['instruction'].strip()}")
    else:
        parts.append(f"Goal: ({row['goal'][0]:.1f}, {row['goal'][1]:.1f}).")
    parts.append(f"Predict the robot's position at each of the next {cfg.horizon} frames.")
    return " ".join(parts)


def kinematic_vector(row):
    """Past positions, current velocity and goal as numbers for the heads (26 values)."""
    ego = row["ego"]
    past = np.stack([ego["x"], ego["y"]], 1)[:CURRENT + 1].ravel()
    goal = (0.0, 0.0) if language_task(row) else row["goal"]  # the end point would leak a language task's answer
    return np.r_[past, ego["vx"][CURRENT], ego["vy"][CURRENT], goal].astype(np.float32)


KIN_DIM = 2 * (CURRENT + 1) + 4


def future_xy(row, cfg):
    """(horizon, 2) recorded future positions in metres."""
    ego = row["ego"]
    xy = np.stack([ego["x"], ego["y"]], 1)[CURRENT + 1:CURRENT + 1 + cfg.horizon]
    return xy.astype(np.float32)


def fit_action_scale(ds, cfg, n=512):
    """Metres per unit so that most targets lie within [-1, 1]: the 95th percentile of |xy|."""
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False)
    sub = ds.select(idx).select_columns(["ego"])
    vals = np.concatenate([np.abs(future_xy({"ego": e}, cfg)).ravel() for e in sub["ego"]])
    return float(max(1.0, np.percentile(vals, 95)))


class NavDataset(Dataset):
    """One item = (sample index, images, prompt, kinematics, trajectory target, auxiliary task targets)."""

    def __init__(self, ds, cfg, tasks=()):
        self.ds, self.cfg, self.tasks = ds, cfg, list(tasks)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        images = row["past_images"][-self.cfg.frames:]
        return dict(index=i, scenario_id=row["scenario_id"], images=[im.convert("RGB") for im in images],
                    prompt=prompt_for(row, self.cfg), kin=kinematic_vector(row),
                    target=future_xy(row, self.cfg) / self.cfg.action_scale,
                    aux={t.name: t.target(row, self.cfg) for t in self.tasks})


# ----------------------------------------------------------------------------- per-frame data (hnod.frames)

def window_config(cfg):
    from hnod.windows import WindowConfig
    return WindowConfig(n_waypoints=cfg.horizon, spacing_m=cfg.spacing_m, n_past=CURRENT, past_dt_s=cfg.past_dt_s,
                        min_indoor_prob=cfg.min_indoor_prob, seed=cfg.seed)


def load_frames(cfg, split):
    """One hnod.windows.FrameWindows per repo in cfg.frames_repos that has this split."""
    from hnod.windows import FrameWindows
    parts = []
    for repo in [r.strip() for r in cfg.frames_repos.split(",") if r.strip()]:
        # name the split's files: load_dataset(repo, "frames", split=...) would download every split first
        try:
            frames = load_dataset(repo, data_files={split: f"data/frames/{split}-*.parquet"}, split=split,
                                  cache_dir=cfg.cache_dir)
        except FileNotFoundError:  # some sources publish only a train split (their held-out data is evaluation data)
            print(f"{repo}: no {split} split, skipped", flush=True)
            continue
        episodes = load_dataset(repo, data_files={split: f"data/episodes/{split}-*.parquet"}, split=split,
                                cache_dir=cfg.cache_dir)
        parts.append(FrameWindows(frames, episodes, window_config(cfg)))
    return parts


def frames_prompt(s, cfg):
    v = np.hypot(*s["velocity"])
    parts = [f"Image interval {cfg.past_dt_s:.1f} s. Current speed {v:.2f} m/s."]
    if cfg.ego_history:
        past = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in s["past_xy"])
        parts.append(f"Robot positions over the last {len(s['past_xy']) - 1} images, oldest first: {past}.")
    parts.append(s["prompt"])
    parts.append(f"Predict {cfg.horizon} waypoints along the robot's path, {cfg.spacing_m:.2f} m apart.")
    return " ".join(parts)


class FrameNavDataset(Dataset):
    """Training items from per-frame data, in the same layout as NavDataset (no auxiliary tasks)."""

    def __init__(self, parts, cfg):
        self.parts, self.cfg = parts, cfg
        self.cum = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, i):
        part = int(np.searchsorted(self.cum, i, side="right") - 1)
        return self.parts[part], i - int(self.cum[part])

    def sample(self, i):
        w, k = self._locate(i)
        return w.sample(k)

    def weights(self, mix):
        """Per-item sampling weights: proportional (all 1) or equal total weight per source."""
        if mix == "equal":
            return np.concatenate([np.full(len(p), 1.0 / max(1, len(p))) for p in self.parts])
        return np.ones(len(self))

    def __getitem__(self, i):
        w, k = self._locate(i)
        s = w[k]
        goal = s["goal"] if s["goal_given"] else np.zeros(2, np.float32)
        kin = np.r_[s["past_xy"].ravel(), s["velocity"], goal].astype(np.float32)
        return dict(index=i, scenario_id=f"{s['episode_id']}:{s['frame_index']}",
                    images=[im.convert("RGB") for im in s["images"][-self.cfg.frames:]],
                    prompt=frames_prompt(s, self.cfg), kin=kin,
                    target=s["target"][:, :self.cfg.action_dim] / self.cfg.action_scale, aux={})


def fit_action_scale_frames(ds, cfg, n=512):
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False)
    vals = np.concatenate([np.abs(ds.sample(int(i))["target"][:, :cfg.action_dim]).ravel() for i in idx])
    return float(max(1.0, np.percentile(vals, 95)))


def columns_for(cfg, tasks):
    cols = list(INPUT_COLUMNS)
    for t in tasks:
        cols += [c for c in t.columns if c not in cols]
    return cols


class Collator:
    """Tokenise prompts with their images (Qwen-VL chat template) and stack targets."""

    def __init__(self, processor, cfg):
        self.processor, self.cfg = processor, cfg
        processor.image_processor.max_pixels = cfg.max_pixels
        processor.image_processor.min_pixels = cfg.min_pixels
        processor.tokenizer.padding_side = "right"  # the last real token of each row is found via the mask

    def __call__(self, items):
        texts, images = [], []
        for it in items:
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [*[{"type": "image"} for _ in it["images"]],
                                                     {"type": "text", "text": it["prompt"]}]}]
            texts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
            images += it["images"]
        batch = dict(self.processor(text=texts, images=images, padding=True, return_tensors="pt"))
        batch["target"] = torch.from_numpy(np.stack([it["target"] for it in items]))
        batch["kin"] = torch.from_numpy(np.stack([it["kin"] for it in items]))
        batch["aux"] = {k: torch.from_numpy(np.stack([it["aux"][k] for it in items])) for k in items[0]["aux"]}
        batch["index"] = torch.tensor([it["index"] for it in items])
        batch["scenario_id"] = [it["scenario_id"] for it in items]
        return batch


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out
