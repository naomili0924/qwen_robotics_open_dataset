#!/usr/bin/env python
"""Convert MuSoHu recordings (ROS1 bags) into scenario Parquet files for the evaluation suite.

MuSoHu has no object labels: moving people are tracked in the lidar (hnod.lidar_tracks); static
obstacles come from the lidar map as for every other source.  All scenarios go to the `test` split:
the whole source is reserved for evaluation.

    python scripts/convert_musohu.py --bags /dev/shm/musohu/bags --out /dev/shm/musohu/hf --repo \
        Jinyan0924/qwen_robotics_open_dataset_musohu --watch
"""
import argparse
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import lidar_bev, lidar_tracks, musohu, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import segment_rows, write_shards  # noqa: E402

CONFIG = "musohu_2hz"
STEP, STRIDE = 5, 10          # 2 Hz steps (5 s + 5 s), one scenario per second
MAP_CONTEXT_FRAMES = 50       # +-5 s of sweeps go into static_map
MIN_STATIC_SPAN = 20          # 2 s
CLOSE_CELLS = 11              # 16-beam lidar: wide gaps between rings
MIN_FUTURE_DISTANCE = 1.0


def camera_inputs(seg, sc):
    """Images and per-image camera poses: the helmet camera tilts relative to the levelled frame."""
    idx = sc["segment_index"][:sc["current_time_index"] + 1]
    paths = [seg["image_paths"][i] for i in idx]
    if any(p is None or not Path(p).exists() for p in paths):
        return None, None
    cam = seg["camera"]
    scenario_from_world = np.linalg.inv(sc["world_from_scenario"])
    poses = [(scenario_from_world @ seg["camera_T"][i]).reshape(-1).tolist() for i in idx]
    return [Path(p).read_bytes() for p in paths], dict(
        name=cam["name"], width=cam["width"], height=cam["height"], K=cam["K"].reshape(-1).tolist(),
        T_scenario_from_camera=poses)


def convert_bag(bag, work):
    name = Path(bag).name.removesuffix(".bag").replace(" ", "_")
    d = musohu.read_bag(bag, Path(work) / "img" / name)
    F = len(d["t"])
    _, yaw = musohu.level(d["R"])  # heading from the odometry
    Rf, height, floor_seen = musohu.level_by_floor(d["sweeps"], d["R"])  # tilt from the floor the lidar sees
    lev = [s @ Rf[i].T for i, s in enumerate(d["sweeps"])]  # levelled sensor frame
    del d["sweeps"]
    T = np.tile(np.eye(4), (F, 1, 1))  # yaw-only pose of the walker's head: the ego and lidar frame
    c, s = np.cos(yaw), np.sin(yaw)
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = c, -s, s, c
    T[:, :3, 3] = d["p"]
    full = np.tile(np.eye(4), (F, 1, 1))
    full[:, :3, :3], full[:, :3, 3] = d["R"], d["p"]
    camera_T = full @ musohu.BASE_FROM_OPTICAL  # world <- camera optical frame, per frame
    bev, store, frames = {}, {}, []
    for i in range(F):
        idx, fi, fj, h, origin = lidar_bev.point_heights(lev[i], T[i, :3, :3], T[i, :3, 3], height)
        grid = lidar_bev.rasterize(fi, fj, h)
        bev[(name, i)] = (origin, zlib.compress(grid.tobytes(), 3))
        store[(name, i)] = lidar_bev.pack_points(*lidar_bev.downsample_points(lev[i][idx], h))
        xy = lev[i][idx, :2] @ T[i, :2, :2].T + T[i, :2, 3]
        frames.append((xy, h, T[i, :2, 3]))
    moving = lidar_tracks.moving_points(frames)
    dets = [lidar_tracks.clusters(xy[m], h[m]) for (xy, h, _), m in zip(frames, moving)]
    del frames, lev
    tracks = lidar_tracks.track(dets, 1.0 / musohu.RATE_HZ)
    rows, n_tracks = [], 0
    for k, (a, b) in enumerate(musohu.cut_segments(d["t"], d["odom_gap"])):
        seg_tracks = [dict(frames=t["frames"][(t["frames"] >= a) & (t["frames"] < b)] - a,
                           xy=t["xy"][(t["frames"] >= a) & (t["frames"] < b)]) for t in tracks]
        seg_tracks = [t for t in seg_tracks if len(t["frames"]) >= lidar_tracks.MIN_HITS]
        n_tracks += len(seg_tracks)
        seg = dict(dataset="musohu", sequence=name, segment=k, rate_hz=musohu.RATE_HZ, frames=np.arange(a, b),
                   timestamps=d["t"][a:b], ego_T=T[a:b], labelled=np.ones(b - a, bool), ego_size=musohu.EGO_SIZE,
                   lidar_height=height,
                   tracks=lidar_tracks.segment_tracks(seg_tracks, b - a, T[a:b, 2, 3] - height),
                   camera=d["camera"], camera_T=camera_T[a:b], point_keys=[(name, i) for i in range(a, b)],
                   bev_keys=[(name, i) for i in range(a, b)], image_paths=d["image_paths"][a:b])
        scenario.clean_segment(seg, lambda _: scenario.PEDESTRIAN)
        rows += segment_rows(seg, bev, STEP, STRIDE, MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN, CLOSE_CELLS,
                             points=store, camera=camera_inputs, min_future_distance=MIN_FUTURE_DISTANCE)
    shutil.rmtree(Path(work) / "img" / name, ignore_errors=True)
    return name, rows, dict(frames=F, sensor_height=round(height, 2), floor_seen=round(floor_seen, 2),
                            tracks=n_tracks, scenarios=len(rows),
                            minutes=round((d["t"][-1] - d["t"][0]) / 60, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default="/dev/shm/musohu/bags")
    ap.add_argument("--out", default="/dev/shm/musohu/hf")
    ap.add_argument("--work", default="/dev/shm/musohu/work")
    ap.add_argument("--repo", default="")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--watch", action="store_true", help="keep converting bags as their downloads finish")
    ap.add_argument("--keep-bags", action="store_true")
    args = ap.parse_args()
    api = None
    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    done = set()
    while True:
        ready = sorted(p for p in Path(args.bags).glob("*.bag") if (p.parent / (p.name + ".done")).exists()
                       or args.only)
        ready = [p for p in ready if p.name not in done and (not args.only or p.name in args.only)]
        for bag in ready:
            t0 = time.time()
            try:
                name, rows, info = convert_bag(bag, args.work)
            except Exception as e:  # a broken bag must not stop the others
                print("FAILED", bag.name, repr(e)[:300], flush=True)
                done.add(bag.name)
                continue
            out = Path(args.out) / CONFIG
            n = write_shards(rows, out / "_parts", f"test-{name}", FEATURES, shard_rows=100)
            print(name, info, f"{time.time() - t0:.0f} s", flush=True)
            if api is not None and n:
                for f in sorted((out / "_parts").glob(f"test-{name}-*.parquet")):
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f"data/{CONFIG}/{f.name}", repo_id=args.repo,
                                    repo_type="dataset", commit_message=f"Add {name}")
                    f.unlink()
            done.add(bag.name)
            if not args.keep_bags:
                bag.unlink()
        if not args.watch or (args.only and set(args.only) <= done):
            break
        if not ready:
            pending = list(Path(args.bags).glob("*.part"))
            if not pending:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
