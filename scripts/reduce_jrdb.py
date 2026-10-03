#!/usr/bin/env python
"""Reduce JRDB lidar sweeps to BEV grids + thinned points, and undistort the forward-camera images.

    python scripts/reduce_jrdb.py --raw /dev/shm/hnod/jrdb --odometry data/raw/jrdb_odometry \
        --out /dev/shm/hnod/jrdb_lidar --images-out /dev/shm/hnod/jrdb_images

Both Velodynes are merged in the robot/label frame before ground segmentation
(see hnod/lidar_bev.py); output files have the layout of stream_robosense_lidar.py
(batch_*.pkl grids, points_*.pkl packed points, keyed by (sequence, frame name)).
"""
import argparse
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pypcd4 import PointCloud
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import jrdb, lidar_bev  # noqa: E402

BATCH = 2000


def load_sweep(raw, seq, name):
    parts = []
    for which in ("upper", "lower"):
        f = Path(raw) / f"pointclouds/{which}_velodyne/{seq}/{name}"
        if f.exists():
            pts = PointCloud.from_path(f).numpy(("x", "y", "z")).astype(np.float64)
            parts.append(jrdb.lidar_to_ego(pts[np.isfinite(pts).all(1)], which))
    return np.concatenate(parts) if parts else None


def reduce_frame(args):
    raw, seq, name, T, height, img_src, img_dst, camera = args
    out = None
    pts = load_sweep(raw, seq, name)
    if pts is not None:
        idx, fi, fj, h, origin = lidar_bev.point_heights(pts, T[:3, :3], T[:3, 3], height)
        packed = lidar_bev.pack_points(*lidar_bev.downsample_points(pts[idx], h))
        out = ((origin, zlib.compress(lidar_bev.rasterize(fi, fj, h).tobytes(), 6)), packed)
    if img_src and Path(img_src).exists() and not Path(img_dst).exists():
        img = cv2.undistort(cv2.imread(img_src), camera["K"], camera["distortion"])
        Path(img_dst).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(img_dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return (seq, name), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/dev/shm/hnod/jrdb")
    ap.add_argument("--odometry", default="data/raw/jrdb_odometry_icp", help="output of refine_jrdb_odometry.py")
    ap.add_argument("--out", default="/dev/shm/hnod/jrdb_lidar")
    ap.add_argument("--images-out", default="/dev/shm/hnod/jrdb_images")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob("batch_*.pkl"):
        done.update(pickle.load(open(f, "rb")))
    camera = jrdb.load_camera(args.raw)
    jobs = []
    for seq in jrdb.sequences(args.raw):
        for seg in jrdb.load_segments(args.raw, args.odometry, seq):
            for i, name in enumerate(seg["frame_names"]):
                if (seq, name) in done:
                    continue
                jobs.append((args.raw, seq, name, seg["ego_T"][i], seg["lidar_height"],
                             str(Path(args.raw) / f"images/{jrdb.CAMERA}/{seq}/{Path(name).stem}.jpg"),
                             str(Path(args.images_out) / seg["image_paths"][i]), camera))
    print(f"{len(jobs)} frames to do, {len(done)} done", flush=True)
    grids, points, n_batch = {}, {}, len(list(out.glob("batch_*.pkl")))

    def flush():
        nonlocal grids, points, n_batch
        pickle.dump(points, open(out / f"points_{n_batch:04d}.pkl", "wb"))
        pickle.dump(grids, open(out / f"batch_{n_batch:04d}.pkl", "wb"))
        n_batch += 1
        grids, points = {}, {}

    with ProcessPoolExecutor(args.workers) as ex:
        for key, res in tqdm(ex.map(reduce_frame, jobs, chunksize=8), total=len(jobs), mininterval=20):
            if res is not None:
                grids[key], points[key] = res
            if len(grids) >= BATCH:
                flush()
    if grids:
        flush()


if __name__ == "__main__":
    main()
