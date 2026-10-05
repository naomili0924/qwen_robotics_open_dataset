#!/usr/bin/env python
"""Convert nuScenes key frames (CAM_FRONT + LIDAR_TOP + 3D boxes) into scenario Parquet files.

    python scripts/convert_nuscenes.py --raw /data/nuscenes --version v1.0-trainval --out data/hf_nuscenes

One config, nuscenes_2hz: 5 s past + 5 s future at the 2 Hz key-frame rate.  Splits follow the
official scene lists when the nuscenes-devkit is installed (its `val` becomes our test, 10 % of
`train` our validation); otherwise scenes are split 80 / 10 / 10 by name hash.  Each scene is
reduced and converted in one worker, so nothing but the raw data and the output is kept on disk.

NOT YET RUN ON REAL DATA (see hnod/nuscenes.py).
"""
import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import lidar_bev, nuscenes, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import camera_inputs, segment_rows, write_shards  # noqa: E402

CONFIG = "nuscenes_2hz"
STRIDE = 2                 # key frames between consecutive scenarios (1 s)
MAP_CONTEXT_FRAMES = 10    # +-5 s of sweeps go into static_map
MIN_STATIC_SPAN = 2        # key frames (1 s) an obstacle cell must persist to count as static
CLOSE_CELLS = 7            # 32-beam lidar
SHARD_ROWS = 128
_tables = None


def split_of(scene, official):
    if official:
        if scene in official["val"]:
            return "test"
        return "validation" if zlib.crc32(scene.encode()) % 10 == 0 else "train"
    return {0: "validation", 1: "test"}.get(zlib.crc32(scene.encode()) % 10, "train")


def convert_scene(args):
    scene, out, split = args
    rows = []
    for seg in nuscenes.load_segments(_tables, scene):
        grids, points = {}, {}
        for i, key in enumerate(seg["bev_keys"]):
            if not Path(seg["lidar_paths"][i]).exists():
                continue
            pts = nuscenes.load_sweep(seg["lidar_paths"][i])
            T = seg["lidar_T"][i]
            idx, fi, fj, h, origin = lidar_bev.point_heights(pts, T[:3, :3], T[:3, 3], seg["lidar_heights"][i])
            points[key] = lidar_bev.pack_points(*lidar_bev.downsample_points(pts[idx], h))
            grids[key] = (origin, zlib.compress(lidar_bev.rasterize(fi, fj, h).tobytes(), 6))
        scenario.clean_segment(seg, nuscenes.type_of)
        rows += segment_rows(seg, grids, 1, STRIDE, MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN, CLOSE_CELLS,
                             points=points, camera=camera_inputs)
    write_shards(rows, Path(out) / CONFIG / "_parts" / split, scene, FEATURES, SHARD_ROWS)
    return scene, split, len(rows)


def main():
    global _tables
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="data/hf_nuscenes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-scenes", type=int, default=0)
    args = ap.parse_args()
    _tables = nuscenes.Tables(args.raw, args.version)
    try:
        from nuscenes.utils.splits import create_splits_scenes
        official = {k: set(v) for k, v in create_splits_scenes().items()}
    except ImportError:
        official = None
        print("nuscenes-devkit not installed: splitting scenes 80/10/10 by name hash")
    scenes = _tables.scene_names()
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    counts = {}
    jobs = [(s, args.out, split_of(s, official)) for s in scenes]
    with ProcessPoolExecutor(args.workers) as ex:   # fork: workers inherit the tables
        for scene, split, n in tqdm(ex.map(convert_scene, jobs), total=len(jobs)):
            counts[split] = counts.get(split, 0) + n
    out = Path(args.out) / CONFIG
    for split, n in sorted(counts.items()):
        parts = sorted((out / "_parts" / split).glob("*.parquet"))
        for f in parts:   # names match scripts/sync_parts_hf.py
            f.rename(out / f"{split}-{f.name}")
        print(f"{CONFIG} {split:10s} {n:7d} scenarios in {len(parts)} shards")


if __name__ == "__main__":
    main()
