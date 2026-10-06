#!/usr/bin/env python
"""Index every candidate final-frame sample of the training sources: motion features + a scene embedding.

Pass 1 of the deduplication (docs/final_frame_pretraining.md, "Less data").  Samples are cut one per second
(the sample cutter's stride) from every frames shard; for each one we record the motion of the next 5 s
(speed, turning, stopping) and a CLIP embedding of the current frame, so that pass 2 (select_dedup.py) can
keep motion-diverse samples first and scene-diverse samples second.

    python scripts/build_dedup_index.py --out /dev/shm/dedup/index
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod.environment import IndoorTagger  # noqa: E402
from hnod.windows import FrameWindows, WindowConfig  # noqa: E402
from vla.stream import _Source, read_shard  # noqa: E402

REPOS = ["Jinyan0924/qwen_robotics_open_dataset_egowalk", "Jinyan0924/habitat_hssd_pointgoal_nav_scenarios",
         "Jinyan0924/qwen_robotics_open_dataset_robosense", "Jinyan0924/qwen_robotics_open_dataset"]
HORIZON_S, N_WAY = 5.0, 10


def motion_features(target):
    """target (10, 3): x, y, yaw over 5 s -> dict of plain numbers."""
    xy = np.vstack([[0.0, 0.0], target[:, :2]])
    seg = np.hypot(*np.diff(xy, axis=0).T)                 # metres per 0.5 s step
    speed = seg / (HORIZON_S / N_WAY)
    heading = np.degrees(target[-1, 2])                    # camera yaw change (a walker's head turns, too)
    bearing = np.degrees(np.arctan2(xy[-1, 1], xy[-1, 0]))  # where the path ends, relative to the current heading
    return dict(dist_m=float(seg.sum()), speed_mean=float(speed.mean()), speed_first=float(speed[:2].mean()),
                speed_last=float(speed[-2:].mean()), speed_min=float(speed.min()), heading_deg=float(heading),
                bearing_deg=float(bearing), lateral_m=float(target[-1, 1]),
                path_straightness=float(seg.sum() / max(1e-6, np.hypot(*xy[-1]))))


def bucket(f):
    """Motion class used for the quotas: rare behaviours first."""
    if f["speed_mean"] < 0.15:
        return "standing"
    if f["speed_last"] < 0.15 and f["speed_first"] > 0.3:
        return "stopping"
    if f["speed_first"] < 0.15 and f["speed_last"] > 0.3:
        return "starting"
    h = f["bearing_deg"]  # the path's direction, not the camera's: a walker looks around without turning
    if abs(h) > 45:
        return "sharp_left" if h > 0 else "sharp_right"
    if abs(h) > 15:
        return "turn_left" if h > 0 else "turn_right"
    if f["speed_last"] - f["speed_first"] < -0.3:
        return "slowing"
    if f["speed_last"] - f["speed_first"] > 0.3:
        return "speeding_up"
    if f["path_straightness"] > 1.15:
        return "weaving"  # not straight but ends ahead: sidesteps, avoidance
    return "straight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/dedup/index")
    ap.add_argument("--repos", nargs="*", default=REPOS)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="k/n: this process takes every n-th shard starting at k (run n in parallel)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tagger = IndoorTagger()

    def embed(images):
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), 256):
                x = tagger.proc(images=images[i:i + 256], return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
                f = tagger.model.get_image_features(pixel_values=x)
                f = getattr(f, "pooler_output", f)
                feats.append((f / f.norm(dim=-1, keepdim=True)).half().cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, 768), np.float16)

    for repo in args.repos:
        src = _Source(repo, args.split)
        episodes = {e["episode_id"]: e for e in src.load_episodes()}
        shards = src.frames[:args.max_shards] if args.max_shards else src.frames
        k, n = (int(x) for x in args.part.split("/"))
        shards = shards[k::n]
        for path in shards:
            dest = out / f"{repo.split('/')[1]}__{Path(path).stem}.parquet"
            if dest.exists():
                continue
            ds = read_shard(repo, path)
            rate = next(iter(episodes.values()))["rate_hz"]
            w = FrameWindows(ds, list(episodes.values()),
                             WindowConfig(mode="final_frame", horizon_s=HORIZON_S, n_waypoints=N_WAY, n_past=5,
                                          past_dt_s=1.0, stride=max(1, int(round(rate)))))
            rows = []
            for k in range(len(w)):
                s = w.sample(k)
                f = motion_features(s["target"])
                rows.append(dict(repo=repo, shard=path, episode_id=s["episode_id"], frame_index=s["frame_index"], row=int(s["row"]),
                                 embodiment=s["embodiment"], bucket=bucket(f), **f))
            if not rows:
                continue
            col = ds.select_columns(["image"])
            def load(i):
                im = col[int(i)]["image"].convert("RGB")
                return np.ascontiguousarray(np.asarray(im.resize((224, max(1, int(224 * im.height / im.width))))))
            with ThreadPoolExecutor(args.workers) as pool:
                images = list(pool.map(load, [r["row"] for r in rows]))
            E = embed(images)
            df = pd.DataFrame(rows)
            df["embedding"] = [e.tobytes() for e in E]
            df.to_parquet(dest)
            print(f"{repo.split('/')[1]} {Path(path).stem}: {len(df)} samples, {dict(df['bucket'].value_counts().head(4))}", flush=True)


if __name__ == "__main__":
    main()
