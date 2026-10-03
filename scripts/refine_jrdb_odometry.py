#!/usr/bin/env python
"""Refine JRDB's wheel odometry with lidar scan matching.

The pre-extracted wheel odometry drifts by up to a metre within a few seconds
while the robot drives.  This registers each merged sweep (both Velodynes, in the
robot frame) to a submap of the previous sweeps with point-to-plane ICP,
initialised from the wheel odometry, and chains the results into new poses.  Pedestrians
are cut out before matching so that moving people do not drag the alignment.

    python scripts/refine_jrdb_odometry.py --raw /dev/shm/hnod/jrdb --odometry data/raw/jrdb_odometry \
        --out data/raw/jrdb_odometry_icp

Output has the layout of the input (odometry/<split>/<sequence>.json), so
hnod.jrdb.load_segments can read either.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import jrdb  # noqa: E402

VOXEL = 0.1
MAX_RANGE = 25.0
MAX_CORR = 0.5
SUBMAP = 10   # sweeps kept as the registration target


def sweep(raw, seq, name, labels):
    """Merged sweep in the robot frame, downsampled, pedestrians removed, with normals."""
    pts = []
    for which in ("upper", "lower"):
        f = Path(raw) / f"pointclouds/{which}_velodyne/{seq}/{name}"
        if f.exists():
            from pypcd4 import PointCloud
            p = PointCloud.from_path(f).numpy(("x", "y", "z")).astype(np.float64)
            pts.append(jrdb.lidar_to_ego(p[np.isfinite(p).all(1)], which))
    p = np.concatenate(pts)
    r = np.linalg.norm(p[:, :2], axis=1)
    p = p[(r > 0.9) & (r < MAX_RANGE)]
    for lab in labels.get(name, []):
        b = lab["box"]
        c, s = np.cos(b["rot_z"]), np.sin(b["rot_z"])
        d = p[:, :2] - [b["cx"], b["cy"]]
        lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        p = p[~((np.abs(lx) < b["l"] / 2 + 0.3) & (np.abs(ly) < b["w"] / 2 + 0.3))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).voxel_down_sample(VOXEL)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
    return pc


def refine_sequence(args):
    raw, odometry, out, seq = args
    labels = json.load(open(Path(raw) / f"labels/labels_3d/{seq}.json"))["labels"]
    odom = jrdb.load_odometry(odometry, seq)
    names = sorted(odom)
    poses = {}
    for n in names:
        T = np.eye(4)
        q, p = odom[n]["orientation"], odom[n]["position"]
        T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        T[:3, 3] = (p["x"], p["y"], p["z"])
        poses[n] = T
    refined = {names[0]: poses[names[0]]}
    first = sweep(raw, seq, names[0], labels)
    recent = [first.transform(refined[names[0]].copy())]  # last SUBMAP sweeps in the refined world frame
    fitness = []
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    for a, b in zip(names[:-1], names[1:]):
        cur = sweep(raw, seq, b, labels)
        init = refined[a] @ np.linalg.inv(poses[a]) @ poses[b]  # wheel-odometry guess for frame b, in the refined world
        target = o3d.geometry.PointCloud()
        for pc in recent:
            target += pc
        target = target.voxel_down_sample(VOXEL)
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
        res = o3d.pipelines.registration.registration_icp(cur, target, MAX_CORR, init, est, crit)
        T = res.transformation if res.fitness > 0.3 else init
        fitness.append(res.fitness)
        refined[b] = T
        recent.append(cur.transform(T.copy()))
        recent = recent[-SUBMAP:]
    d = Path(out) / "train"
    d.mkdir(parents=True, exist_ok=True)
    out_odom = {}
    for n, T in refined.items():
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        out_odom[n] = {"position": {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3])},
                       "orientation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}}
    json.dump({"odometry": out_odom}, open(d / f"{seq}.json", "w"))
    return seq, float(np.mean(fitness)), float(np.mean(np.array(fitness) < 0.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/dev/shm/hnod/jrdb")
    ap.add_argument("--odometry", default="data/raw/jrdb_odometry")
    ap.add_argument("--out", default="data/raw/jrdb_odometry_icp")
    ap.add_argument("--sequences", nargs="*")
    ap.add_argument("--workers", type=int, default=27)
    args = ap.parse_args()
    seqs = args.sequences or jrdb.sequences(args.raw)
    with ProcessPoolExecutor(args.workers) as ex:
        for seq, fit, bad in ex.map(refine_sequence, [(args.raw, args.odometry, args.out, s) for s in seqs]):
            print(f"{seq:45s} mean ICP fitness {fit:.2f}, frames falling back to wheel odometry {100 * bad:.1f}%", flush=True)


if __name__ == "__main__":
    main()
