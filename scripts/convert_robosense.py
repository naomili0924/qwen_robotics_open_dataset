#!/usr/bin/env python
"""Convert RoboSense annotations (+ reduced lidar grids) into scenario Parquet files.

Usage:
    python scripts/convert_robosense.py --pkl data/raw/robosense/splits --bev data/interim/robosense_bev --out data/hf_robosense
"""
import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import robosense, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import segment_rows, write_shards  # noqa: E402

CONFIG = "robosense_1hz"
STEP = 1                      # RoboSense is labelled at 1 Hz: 10 s of history, 10 s of future
STRIDE = {"train": 2, "val": 1}  # seconds between consecutive scenarios
SPLIT_NAME = {"train": "train", "val": "validation"}
MAP_CONTEXT_FRAMES = 5        # +-5 s of sweeps (11 at most) go into static_map
MIN_STATIC_SPAN = 2           # frames (2 s) an obstacle cell must persist to count as static
CLOSE_CELLS = 11              # the 64-beam lidar leaves ring gaps of up to ~1 m on open ground
TYPE_OF = {"Car": scenario.VEHICLE, "Pedestrian": scenario.PEDESTRIAN, "Cyclist": scenario.CYCLE}

_bev = {}


def convert_segment(seg):
    scenario.clean_segment(seg, TYPE_OF.__getitem__)
    bev = {k: _bev[k] for k in seg["bev_keys"] if k in _bev}
    return segment_rows(seg, bev, STEP, STRIDE[seg["split"]], MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN, CLOSE_CELLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/raw/robosense/splits")
    ap.add_argument("--bev", default="data/interim/robosense_bev")
    ap.add_argument("--out", default="data/hf_robosense")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--require-lidar", action="store_true", help="skip chains whose sweeps are not all reduced yet")
    args = ap.parse_args()

    for f in sorted(Path(args.bev).glob("batch_*.pkl")):
        _bev.update(pickle.load(open(f, "rb")))
    print(f"{len(_bev)} lidar grids")
    for split in ("val", "train"):
        segs = robosense.load_segments(args.pkl, split, min_frames=scenario.N_STEPS)
        if args.require_lidar:
            segs = [s for s in segs if all(k in _bev for k in s["bev_keys"])]
        rows = []
        with ProcessPoolExecutor(args.workers) as ex:  # fork: workers inherit _bev
            for r in tqdm(ex.map(convert_segment, segs, chunksize=4), total=len(segs), desc=split):
                rows += r
        n = write_shards(rows, Path(args.out) / CONFIG, SPLIT_NAME[split], FEATURES)
        print(f"{CONFIG} {SPLIT_NAME[split]:10s} {len(rows):6d} scenarios from {len(segs)} chains in {n} shards")


if __name__ == "__main__":
    main()
