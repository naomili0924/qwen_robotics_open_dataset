#!/usr/bin/env python
"""Convert CODa annotations (+ reduced lidar grids) into scenario Parquet files.

Usage:
    python scripts/convert_coda.py --raw data/raw/coda --bev data/interim/coda_bev --out data/hf
"""
import argparse
import collections
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import coda, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import segment_rows, write_shards  # noqa: E402

# name -> (frames between steps, frames between consecutive scenarios) at CODa's 10 Hz
CONFIGS = {"coda_10hz": (1, 5), "coda_2hz": (5, 10)}
# Split by whole sequence so no place-and-time is shared between splits.
SPLIT_OF = {**{s: "validation" for s in (4, 10, 19)}, **{s: "test" for s in (1, 7, 9, 12, 17)}}
MAP_CONTEXT_FRAMES = 50  # static_map merges lidar from +-5 s around the current step, observed_map the past 5 s
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


def group_bev_by_sequence(bev_dir):
    """Regroup the streaming batches into one {frame: (origin, blob)} pickle per sequence."""
    bev_dir = Path(bev_dir)
    if not list(bev_dir.glob("seq_*.pkl")):
        by_seq = collections.defaultdict(dict)
        for f in sorted(bev_dir.glob("batch_*.pkl")):
            with open(f, "rb") as fh:
                for seq, frame, origin, blob in pickle.load(fh)["frames"]:
                    by_seq[seq][frame] = (origin, blob)
        for seq, d in by_seq.items():
            with open(bev_dir / f"seq_{seq}.pkl", "wb") as fh:
                pickle.dump(d, fh)


def convert_sequence(args):
    raw, bev_dir, seq = args
    bev_file = Path(bev_dir) / f"seq_{seq}.pkl"
    bev = pickle.load(open(bev_file, "rb")) if bev_file.exists() else {}
    rows = {name: [] for name in CONFIGS}
    for seg in coda.load_segments(raw, seq):
        scenario.clean_segment(seg, type_of)
        for name, (step, stride) in CONFIGS.items():
            rows[name] += segment_rows(seg, bev, step, stride, MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN)
    return seq, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/coda")
    ap.add_argument("--bev", default="data/interim/coda_bev")
    ap.add_argument("--out", default="data/hf")
    ap.add_argument("--workers", type=int, default=21)
    ap.add_argument("--sequences", type=int, nargs="*")
    args = ap.parse_args()

    group_bev_by_sequence(args.bev)
    seqs = args.sequences or coda.sequences(args.raw)
    rows = {name: collections.defaultdict(list) for name in CONFIGS}
    with ProcessPoolExecutor(args.workers) as ex:
        jobs = [(args.raw, args.bev, s) for s in seqs]
        for seq, out in tqdm(ex.map(convert_sequence, jobs), total=len(jobs)):
            for name, r in out.items():
                rows[name][SPLIT_OF.get(seq, "train")] += r

    for name, splits in rows.items():
        for split, r in splits.items():
            n = write_shards(r, Path(args.out) / name, split, FEATURES)
            print(f"{name:10s} {split:10s} {len(r):6d} scenarios in {n} shards")


if __name__ == "__main__":
    main()
