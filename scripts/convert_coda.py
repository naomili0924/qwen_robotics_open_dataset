#!/usr/bin/env python
"""Convert CODa annotations, front-camera images and reduced lidar into scenario Parquet files.

Usage:
    python scripts/convert_coda.py --raw data/raw/coda --bev data/interim/coda_lidar \
        --images data/interim/coda_images --out data/hf
"""
import argparse
import collections
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import coda, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import camera_inputs, segment_rows, write_shards  # noqa: E402

# name -> (frames between steps, frames between consecutive scenarios) at CODa's 10 Hz
CONFIGS = {"coda_10hz": (1, 10), "coda_2hz": (5, 10)}
SHARD_ROWS = 64          # a scenario is ~6 MB with its 11 images and 11 point clouds
# Split by whole sequence so no place-and-time is shared between splits.
SPLIT_OF = {**{s: "validation" for s in (4, 10, 19)}, **{s: "test" for s in (1, 7, 9, 12, 17)}}
MAP_CONTEXT_FRAMES = 50  # static_map merges lidar from +-5 s around the current step
MIN_STATIC_SPAN = 20     # frames (2 s) an obstacle cell must persist to count as static

_TYPES = {
    scenario.PEDESTRIAN: {"Pedestrian"},
    scenario.CYCLE: {"Bike", "Scooter", "Motorcycle", "Skateboard", "Segway"},
    scenario.VEHICLE: {"Car", "Service Vehicle", "Pickup Truck", "Utility Vehicle", "Delivery Truck", "Bus"},
    scenario.OTHER_MOVABLE: {"Cart", "Horse"},
}
_TYPE_OF = {c: t for t, cs in _TYPES.items() for c in cs}


def type_of(category):
    return _TYPE_OF.get(category, scenario.STATIC)


def group_by_sequence(bev_dir):
    """Regroup the streaming batches into one pickle per sequence.

    seq_<n>.pkl: {frame: (origin, grid blob)}, points_seq_<n>.pkl: {(seq, frame): packed points}.
    """
    bev_dir = Path(bev_dir)
    if list(bev_dir.glob("seq_*.pkl")):
        return
    grids, points = collections.defaultdict(dict), collections.defaultdict(dict)
    for f in sorted(bev_dir.glob("batch_*.pkl")):
        for seq, frame, origin, blob in pickle.load(open(f, "rb"))["frames"]:
            grids[seq][frame] = (origin, blob)
    for f in sorted(bev_dir.glob("points_[0-9]*.pkl")):
        for key, packed in pickle.load(open(f, "rb")).items():
            points[key[0]][key] = packed
    for seq, d in grids.items():
        pickle.dump(d, open(bev_dir / f"seq_{seq}.pkl", "wb"))
        pickle.dump(points[seq], open(bev_dir / f"points_seq_{seq}.pkl", "wb"))


def convert_sequence(args):
    raw, bev_dir, images, out, seq = args
    bev_dir = Path(bev_dir)
    bev = pickle.load(open(bev_dir / f"seq_{seq}.pkl", "rb"))
    points = pickle.load(open(bev_dir / f"points_seq_{seq}.pkl", "rb"))
    camera = coda.load_camera(raw, seq)
    rows = {name: [] for name in CONFIGS}
    for seg in coda.load_segments(raw, seq):
        scenario.clean_segment(seg, type_of)
        seg["camera"] = camera
        seg["image_paths"] = [f"{images}/{seq}/{int(f)}.jpg" for f in seg["frames"]]
        seg["point_keys"] = [(seq, int(f)) for f in seg["frames"]]
        for name, (step, stride) in CONFIGS.items():
            rows[name] += segment_rows(seg, bev, step, stride, MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN,
                                       points=points, camera=camera_inputs)
    # Shards are written by the worker: a sequence's rows hold gigabytes of images and points.
    split = SPLIT_OF.get(seq, "train")
    for name, r in rows.items():
        write_shards(r, Path(out) / name / "_parts" / split, f"seq{seq:02d}", FEATURES, SHARD_ROWS)
    return seq, split, {name: len(r) for name, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/coda")
    ap.add_argument("--bev", default="data/interim/coda_lidar", help="output of stream_coda_lidar.py")
    ap.add_argument("--images", default="data/interim/coda_images", help="output of stream_coda_images.py")
    ap.add_argument("--out", default="data/hf")
    ap.add_argument("--workers", type=int, default=21)
    ap.add_argument("--sequences", type=int, nargs="*")
    args = ap.parse_args()

    group_by_sequence(args.bev)
    seqs = args.sequences or coda.sequences(args.raw)
    counts = collections.Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        jobs = [(args.raw, args.bev, args.images, args.out, s) for s in seqs]
        for seq, split, n in tqdm(ex.map(convert_sequence, jobs), total=len(jobs)):
            for name, k in n.items():
                counts[name, split] += k

    # Give the per-sequence part files their final <split>-NNNNN-of-NNNNN names.
    for (name, split), n_rows in sorted(counts.items()):
        parts = sorted((Path(args.out) / name / "_parts" / split).glob("*.parquet"))
        for k, f in enumerate(parts):
            f.rename(Path(args.out) / name / f"{split}-{k:05d}-of-{len(parts):05d}.parquet")
        print(f"{name:10s} {split:10s} {n_rows:6d} scenarios in {len(parts)} shards")
    for name in CONFIGS:
        shutil.rmtree(Path(args.out) / name / "_parts", ignore_errors=True)


if __name__ == "__main__":
    main()
