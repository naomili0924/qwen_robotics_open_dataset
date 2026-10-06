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


FINAL_FRAME_SYSTEM = ("You watch a camera carried through the world by a human or a robot. The images are its front "
                      "view: first the past, oldest first, then the view now, then the view at the end of the next "
                      "{h:g} seconds. Predict where the camera is at {n} moments evenly spaced over those {h:g} seconds, "
                      "in metres relative to its position now: x forward, y left.")


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
    if cfg.window_mode == "final_frame":
        return WindowConfig(mode="final_frame", horizon_s=cfg.horizon_s, n_waypoints=cfg.horizon,
                            n_past=cfg.frames - 2, past_dt_s=cfg.past_dt_s, min_indoor_prob=cfg.min_indoor_prob,
                            seed=cfg.seed)
    return WindowConfig(n_waypoints=cfg.horizon, spacing_m=cfg.spacing_m, n_past=CURRENT, past_dt_s=cfg.past_dt_s,
                        min_indoor_prob=cfg.min_indoor_prob, seed=cfg.seed)


def load_frames(cfg, split):
    """One hnod.windows.FrameWindows per repo in cfg.frames_repos that has this split."""
    from hnod.windows import FrameWindows
    parts = []
    for repo in [r.strip() for r in cfg.frames_repos.split(",") if r.strip()]:
        # a Hub repo, or a local copy made by scripts/cache_frames.py (<dir>/frames, <dir>/episodes)
        local = os.path.isdir(repo)
        fr = sorted(glob.glob(f"{repo}/frames/{split}-*.parquet")) if local else f"data/frames/{split}-*.parquet"
        ep = sorted(glob.glob(f"{repo}/episodes/{split}-*.parquet")) if local else f"data/episodes/{split}-*.parquet"
        if local and not fr:
            print(f"{repo}: no {split} split, skipped", flush=True)
            continue
        try:  # name the split's files: load_dataset(repo, "frames", split=...) would download every split first
            frames = load_dataset("parquet" if local else repo, data_files={split: fr}, split=split,
                                  cache_dir=cfg.cache_dir)
        except FileNotFoundError:  # some sources publish only a train split (their held-out data is evaluation data)
            print(f"{repo}: no {split} split, skipped", flush=True)
            continue
        episodes = load_dataset("parquet" if local else repo, data_files={split: ep}, split=split,
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


def final_frame_tags(n_images, cfg):
    """Text placed before each image: past views, the current view, the view at the end of the horizon."""
    n_past = n_images - 2
    past = [f"View {cfg.past_dt_s * (n_past - j):g} s ago:" for j in range(n_past)]
    return past + ["View now:", f"View in {cfg.horizon_s:g} s:"]


def final_frame_item(images, final_image, prompt, cfg, target=None, index=0, scenario_id="0"):
    """Item for the final-frame task: images + embodiment sentence only (no positions, speed or goal numbers)."""
    imgs = [im.convert("RGB") for im in images][-(cfg.frames - 1):]
    imgs = [imgs[0]] * (cfg.frames - 1 - len(imgs)) + imgs + [final_image.convert("RGB")]
    if target is None:
        target = np.zeros((cfg.horizon, cfg.action_dim), np.float32)
    return dict(index=index, scenario_id=scenario_id, images=imgs, image_tags=final_frame_tags(len(imgs), cfg),
                system=FINAL_FRAME_SYSTEM.format(h=cfg.horizon_s, n=cfg.horizon), prompt=prompt,
                kin=np.zeros(KIN_DIM, np.float32), target=np.asarray(target, np.float32), aux={})


def frame_item(windows, k, cfg, index=0):
    """One training item (NavDataset layout) for sample k of a hnod.windows.FrameWindows."""
    s = windows[k]
    if "final_image" in s:
        item = final_frame_item(s["images"], s["final_image"], s["prompt"], cfg,
                                target=s["target"][:, :cfg.action_dim] / cfg.action_scale, index=index,
                                scenario_id=f"{s['episode_id']}:{s['frame_index']}")
        if cfg.text_loss:  # the sample's description (hindsight text) is a training target, never an input
            item["description"] = getattr(windows, "descriptions", {}).get((s["episode_id"], s["frame_index"]), "")
        return item
    goal = s["goal"] if s["goal_given"] else np.zeros(2, np.float32)
    kin = np.r_[s["past_xy"].ravel(), s["velocity"], goal].astype(np.float32)
    return dict(index=index, scenario_id=f"{s['episode_id']}:{s['frame_index']}",
                images=[im.convert("RGB") for im in s["images"][-cfg.frames:]],
                prompt=frames_prompt(s, cfg), kin=kin,
                target=s["target"][:, :cfg.action_dim] / cfg.action_scale, aux={})


class _StreamItem:
    """Picklable item function for vla.stream.StreamFrames."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, windows, k):
        return frame_item(windows, k, self.cfg)


def stream_frames(cfg, split, finite=False, max_items=0):
    """vla.stream.StreamFrames over cfg.frames_repos (Hub repos), bounded disk use."""
    import dataclasses
    from vla.stream import StreamFrames
    wcfg = window_config(cfg)
    if finite:
        wcfg = dataclasses.replace(wcfg, stride=10)  # validation: spread a fixed number of items over episodes
    repos = [r.strip() for r in cfg.frames_repos.split(",") if r.strip()]
    return StreamFrames(repos, split, wcfg, _StreamItem(cfg), mix=cfg.frames_mix, seed=cfg.seed, finite=finite,
                        max_items=max_items)


def fit_action_scale_stream(stream, cfg, per_source=256):
    """Action scale from one shard of every source (the 95th percentile of |target|)."""
    from vla.stream import read_shard
    from hnod.windows import FrameWindows
    vals = []
    rng = np.random.default_rng(0)
    for src in stream.sources:
        w = FrameWindows(read_shard(src.repo, src.frames[0]), stream.episodes[src.repo], stream.window_cfg)
        for k in rng.choice(len(w), size=min(per_source, len(w)), replace=False):
            vals.append(np.abs(w.sample(int(k))["target"][:, :cfg.action_dim]).ravel())
    return float(max(1.0, np.percentile(np.concatenate(vals), 95)))


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
        if mix == "sqrt":
            return np.concatenate([np.full(len(p), 1.0 / np.sqrt(max(1, len(p)))) for p in self.parts])
        return np.ones(len(self))

    def __getitem__(self, i):
        w, k = self._locate(i)
        return frame_item(w, k, self.cfg, index=i)


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

    def render(self, messages):
        """Chat-template text; a base model without a template gets the same turns written out by hand
        (Qwen's <|im_start|> format, images as <|vision_start|><|image_pad|><|vision_end|>)."""
        if getattr(self.processor, "chat_template", None):
            return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tok = getattr(self.processor, "image_token", "<|image_pad|>")
        out = []
        for m in messages:
            body = "".join(f"<|vision_start|>{tok}<|vision_end|>" if c["type"] == "image" else c["text"] for c in m["content"])
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(out) + "<|im_start|>assistant\n"

    def __call__(self, items):
        texts, images = [], []
        for it in items:
            tags = it.get("image_tags")
            views = [{"type": "image"} for _ in it["images"]] if not tags else \
                [part for tag in tags for part in ({"type": "text", "text": tag}, {"type": "image"})]
            messages = [{"role": "system", "content": [{"type": "text", "text": it.get("system", SYSTEM_PROMPT)}]},
                        {"role": "user", "content": [*views, {"type": "text", "text": it["prompt"]}]}]
            texts.append(self.render(messages))
            images += it["images"]
        batch = dict(self.processor(text=texts, images=images, padding=True, return_tensors="pt"))
        if self.cfg.text_loss and any(it.get("description") for it in items):
            batch = self.append_text(batch, [it.get("description", "") for it in items])
        batch["target"] = torch.from_numpy(np.stack([it["target"] for it in items]))
        batch["kin"] = torch.from_numpy(np.stack([it["kin"] for it in items]))
        batch["aux"] = {k: torch.from_numpy(np.stack([it["aux"][k] for it in items])) for k in items[0]["aux"]}
        batch["index"] = torch.tensor([it["index"] for it in items])
        batch["scenario_id"] = [it["scenario_id"] for it in items]
        return batch


    def append_text(self, batch, texts):
        """Write each description after the prompt (assistant turn) with labels on its tokens only; the heads keep
        reading the last prompt token (`summary_pos`), which never sees the text under causal attention."""
        tok = self.processor.tokenizer
        end = tok.convert_tokens_to_ids("<|im_end|>")
        lens = batch["attention_mask"].sum(1).tolist()
        extra = [(tok(t, add_special_tokens=False)["input_ids"] + [end]) if t else [] for t in texts]
        L = max(n + len(e) for n, e in zip(lens, extra))
        B = len(lens)
        ids = torch.full((B, L), tok.pad_token_id, dtype=batch["input_ids"].dtype)
        mask = torch.zeros((B, L), dtype=batch["attention_mask"].dtype)
        labels = torch.full((B, L), -100, dtype=torch.long)
        types = torch.zeros((B, L), dtype=batch["mm_token_type_ids"].dtype) if "mm_token_type_ids" in batch else None
        for i, (n, e) in enumerate(zip(lens, extra)):
            ids[i, :n] = batch["input_ids"][i, :n]
            mask[i, :n + len(e)] = 1
            if types is not None:
                types[i, :n] = batch["mm_token_type_ids"][i, :n]
            if e:
                ids[i, n:n + len(e)] = torch.tensor(e)
                labels[i, n:n + len(e)] = torch.tensor(e)
        batch.update(input_ids=ids, attention_mask=mask, labels=labels, summary_pos=torch.tensor(lens) - 1)
        if types is not None:
            batch["mm_token_type_ids"] = types
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
