#!/usr/bin/env python
"""Convert JRDB 3D pedestrian labels + odometry into scenario Parquet files.

STATUS: untested on real JRDB labels - see hnod/jrdb.py.  Produces tracks only
(JRDB annotates pedestrians but no static objects, and this script adds neither
images nor lidar), so `past_images`, `future_lidar` and `static_map` are empty
and only agent collisions can be evaluated.

Usage:
    python scripts/convert_jrdb.py --labels <jrdb>/train/labels/labels_3d \
        --odometry <odometry>/train --out data/hf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import jrdb, scenario  # noqa: E402
from hnod.io import FEATURES, to_row  # noqa: E402

# name -> (frames between steps, frames between consecutive scenarios) at JRDB's 15 Hz
CONFIGS = {"jrdb_7.5hz": (2, 8), "jrdb_2.5hz": (6, 15)}
SHARD_ROWS = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--odometry", required=True)
    ap.add_argument("--out", default="data/hf")
    args = ap.parse_args()

    rows = {name: [] for name in CONFIGS}
    for seq in jrdb.sequences(args.labels):
        seg = scenario.clean_segment(jrdb.load_segment(args.labels, args.odometry, seq), lambda c: scenario.PEDESTRIAN)
        for name, (step, stride) in CONFIGS.items():
            for a in scenario.window_anchors(seg, step, stride):
                rows[name].append(to_row(scenario.build_scenario(seg, a, step), None, float("nan")))
        print(seq, {n: len(r) for n, r in rows.items()})

    for name, r in rows.items():
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        n = int(np.ceil(len(r) / SHARD_ROWS))
        for k in range(n):
            Dataset.from_list(r[k * SHARD_ROWS:(k + 1) * SHARD_ROWS], features=FEATURES) \
                .to_parquet(d / f"train-{k:05d}-of-{n:05d}.parquet")
        print(f"{name}: {len(r)} scenarios in {n} shards")


if __name__ == "__main__":
    main()
