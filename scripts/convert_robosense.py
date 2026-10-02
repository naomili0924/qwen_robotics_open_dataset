#!/usr/bin/env python
"""Convert RoboSense annotations, front-camera images and reduced lidar into scenario Parquet files.

Usage:
    python scripts/convert_robosense.py --pkl data/raw/robosense/splits --bev data/interim/robosense_lidar \
        --images data/interim/robosense_images --out data/hf_robosense
"""
import argparse
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import robosense, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import camera_inputs, segment_rows, write_shards  # noqa: E402

CONFIG = "robosense_1hz"
STEP = 1                      # RoboSense is labelled at 1 Hz: 10 s of history, 10 s of future
STRIDE = {"train": 4, "val": 1}  # seconds between consecutive scenarios
SPLIT_NAME = {"train": "train", "val": "validation"}
MAP_CONTEXT_FRAMES = 5        # +-5 s of sweeps (11 at most) go into static_map
MIN_STATIC_SPAN = 2           # frames (2 s) an obstacle cell must persist to count as static
CLOSE_CELLS = 11              # the 64-beam lidar leaves ring gaps of up to ~1 m on open ground
SHARD_ROWS = 64               # a scenario is several MB with its 11 images and 11 point clouds
CHAINS_PER_JOB = 12
TYPE_OF = {"Car": scenario.VEHICLE, "Pedestrian": scenario.PEDESTRIAN, "Cyclist": scenario.CYCLE}

_bev, _points = {}, {}


def convert_job(args):
    job, segs, images, out = args
    rows = []
    for seg in segs:
        scenario.clean_segment(seg, TYPE_OF.__getitem__)
        seg["image_paths"] = [images + p for p in seg["image_paths"]]
        bev = {k: _bev[k] for k in seg["bev_keys"] if k in _bev}
        rows += segment_rows(seg, bev, STEP, STRIDE[seg["split"]], MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN, CLOSE_CELLS,
                             points=_points, camera=camera_inputs)
    # Shards are written by the worker: the rows hold gigabytes of images and points.
    write_shards(rows, out, f"job{job:04d}", FEATURES, SHARD_ROWS)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/raw/robosense/splits")
    ap.add_argument("--bev", default="data/interim/robosense_lidar", help="output of stream_robosense_lidar.py")
    ap.add_argument("--images", default="data/interim/robosense_images", help="output of stream_robosense_images.py")
    ap.add_argument("--out", default="data/hf_robosense")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    for f in sorted(Path(args.bev).glob("batch_*.pkl")):
        _bev.update(pickle.load(open(f, "rb")))
    for f in sorted(Path(args.bev).glob("points_*.pkl")):
        _points.update(pickle.load(open(f, "rb")))
    print(f"{len(_bev)} lidar grids, {len(_points)} point clouds")
    for split in ("val", "train"):
        segs = robosense.load_segments(args.pkl, split, min_frames=scenario.N_STEPS)
        parts = Path(args.out) / CONFIG / "_parts" / split
        jobs = [(j, segs[i:i + CHAINS_PER_JOB], args.images, parts)
                for j, i in enumerate(range(0, len(segs), CHAINS_PER_JOB))]
        with ProcessPoolExecutor(args.workers) as ex:  # fork: workers inherit _bev and _points
            n_rows = sum(tqdm(ex.map(convert_job, jobs), total=len(jobs), desc=split))
        files = sorted(parts.glob("*.parquet"))
        for k, f in enumerate(files):
            f.rename(Path(args.out) / CONFIG / f"{SPLIT_NAME[split]}-{k:05d}-of-{len(files):05d}.parquet")
        print(f"{CONFIG} {SPLIT_NAME[split]:10s} {n_rows:6d} scenarios from {len(segs)} chains in {len(files)} shards")
    shutil.rmtree(Path(args.out) / CONFIG / "_parts", ignore_errors=True)


if __name__ == "__main__":
    main()
