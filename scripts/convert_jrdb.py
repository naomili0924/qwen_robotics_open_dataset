#!/usr/bin/env python
"""Convert JRDB labels, forward-camera images and reduced lidar into scenario Parquet files.

    python scripts/convert_jrdb.py --raw /dev/shm/hnod/jrdb --odometry data/raw/jrdb_odometry \
        --bev /dev/shm/hnod/jrdb_lidar --images /dev/shm/hnod/jrdb_images --out /dev/shm/hnod/hf_jrdb
"""
import argparse
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import jrdb, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import camera_inputs, segment_rows, write_shards  # noqa: E402

# name -> (frames between steps, frames between consecutive scenarios) at JRDB's 15 Hz
CONFIGS = {"jrdb_2.5hz": (6, 15), "jrdb_5hz": (3, 15)}
SHARD_ROWS = 128
MAP_CONTEXT_FRAMES = 75   # +-5 s of sweeps go into static_map
MIN_STATIC_SPAN = 30      # frames (2 s) an obstacle cell must persist to count as static
CLOSE_CELLS = 11          # two 16-beam lidars leave wide ring gaps
MIN_FUTURE_DISTANCE = 1.0  # the robot stands still in 13 of 27 sequences; keep scenarios where it moves
# Splits by whole sequence; the first group has the robot moving, the second standing.
SPLIT_OF = {
    "clark-center-2019-02-28_1": "test", "huang-basement-2019-01-25_0": "test",
    "packard-poster-session-2019-03-20_2": "test", "tressider-2019-03-16_1": "test",
    "gates-to-clark-2019-02-28_1": "validation", "bytes-cafe-2019-02-07_0": "validation",
}

_bev, _points = {}, {}


def convert_sequence(args):
    raw, odometry, images, out, seq = args
    rows = {name: [] for name in CONFIGS}
    for seg in jrdb.load_segments(raw, odometry, seq):
        scenario.clean_segment(seg, lambda c: scenario.PEDESTRIAN)
        seg["image_paths"] = [f"{images}/{p}" for p in seg["image_paths"]]
        bev = {k: _bev[k] for k in seg["bev_keys"] if k in _bev}
        for name, (step, stride) in CONFIGS.items():
            rows[name] += segment_rows(seg, bev, step, stride, MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN, CLOSE_CELLS,
                                       points=_points, camera=camera_inputs, min_future_distance=MIN_FUTURE_DISTANCE)
    split = SPLIT_OF.get(seq, "train")
    for name, r in rows.items():
        write_shards(r, Path(out) / name / "_parts" / split, seq, FEATURES, SHARD_ROWS)
    return seq, split, {name: len(r) for name, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/dev/shm/hnod/jrdb")
    ap.add_argument("--odometry", default="data/raw/jrdb_odometry_icp", help="output of refine_jrdb_odometry.py")
    ap.add_argument("--bev", default="/dev/shm/hnod/jrdb_lidar")
    ap.add_argument("--images", default="/dev/shm/hnod/jrdb_images")
    ap.add_argument("--out", default="data/hf_jrdb")
    ap.add_argument("--workers", type=int, default=27)
    args = ap.parse_args()

    for f in sorted(Path(args.bev).glob("batch_*.pkl")):
        _bev.update(pickle.load(open(f, "rb")))
    for f in sorted(Path(args.bev).glob("points_*.pkl")):
        _points.update(pickle.load(open(f, "rb")))
    print(f"{len(_bev)} lidar grids, {len(_points)} point clouds")
    seqs = jrdb.sequences(args.raw)
    counts = {}
    with ProcessPoolExecutor(args.workers) as ex:  # fork: workers inherit _bev / _points
        jobs = [(args.raw, args.odometry, args.images, args.out, s) for s in seqs]
        for seq, split, n in tqdm(ex.map(convert_sequence, jobs), total=len(jobs)):
            for name, k in n.items():
                counts[name, split] = counts.get((name, split), 0) + k
    for (name, split), n_rows in sorted(counts.items()):
        parts = sorted((Path(args.out) / name / "_parts" / split).glob("*.parquet"))
        for k, f in enumerate(parts):
            f.rename(Path(args.out) / name / f"{split}-{k:05d}-of-{len(parts):05d}.parquet")
        print(f"{name:10s} {split:10s} {n_rows:6d} scenarios in {len(parts)} shards")
    for name in CONFIGS:
        shutil.rmtree(Path(args.out) / name / "_parts", ignore_errors=True)


if __name__ == "__main__":
    main()
