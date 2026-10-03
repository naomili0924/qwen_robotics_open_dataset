"""Reader for JRDB (JackRabbot Dataset) 3D pedestrian labels, odometry, timestamps and calibration.

Layout (the train archive, `train_dataset_with_activity/`):
  labels/labels_3d/<sequence>.json       {"labels": {"000000.pcd": [{"label_id": "pedestrian:12",
                                           "box": {cx, cy, cz, l, w, h, rot_z}, "attributes": {...}}, ...]}}
  timestamps/<sequence>/frames_pc.json   per frame: lidar timestamps
  calibration/cameras.yaml, defaults.yaml
  images/image_0/<sequence>/NNNNNN.jpg   forward camera (optical axis = +x), 752 x 480, distorted
  pointclouds/{upper,lower}_velodyne/<sequence>/NNNNNN.pcd
Odometry is not in the archive; the pre-extracted copy published with Google's
Human Scene Transformer is used (odometry/<split>/<sequence>.json, one pose per
frame, z = 0): https://storage.googleapis.com/gresearch/human_scene_transformer/odometry.zip

Conventions, checked by projecting boxes and lidar into the images:
  * boxes are in the robot frame centred on the camera rig: x forward, y left, z up;
    (cx, cy, cz) is the box centre; the ground is ~0.76 m below the origin
  * the upper lidar sits 0.49 m above the label-frame origin and is yawed by 0.085 rad,
    the lower lidar 0.02 m above it (see UPPER_LIDAR / LOWER_LIDAR for how this was measured)
  * camera i: p_cam = R_i @ A @ p + T_i / 1000 with A mapping (x, y, z) -> (-y, -z, x)
"""
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

RATE_HZ = 15.0
MAX_STEP_S = 0.2          # frames further apart than this start a new segment
# JackRabbot (Segway RMP210 base) approximate footprint and height [m].
EGO_SIZE = (0.65, 0.55, 1.2)
CAMERA = "image_0"
_AXES = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # robot (x fwd, y left, z up) -> optical (x right, y down, z fwd)
# Lidar -> label frame.  The yaw comes from calibration/defaults.yaml; the vertical offsets
# do not: with the file's values the two sweeps disagree by 0.95 m.  These were measured by
# fitting the floor plane in each lidar's own frame (upper -1.24 m, lower -0.77 m) and
# placing it where the labels put it (box bottoms at -0.75 m).
UPPER_LIDAR = dict(yaw=0.085, translation=(0.0, 0.0, 0.49))
LOWER_LIDAR = dict(yaw=0.0, translation=(0.0, 0.0, 0.02))


def sequences(raw):
    return sorted(p.stem for p in (Path(raw) / "labels/labels_3d").glob("*.json"))


def load_timestamps(raw, seq):
    data = json.load(open(Path(raw) / f"timestamps/{seq}/frames_pc.json"))["data"]
    return {Path(fr["pointclouds"][0]["url"]).name: float(fr["pointclouds"][0]["timestamp"]) for fr in data}


def load_odometry(odometry_dir, seq):
    for split in ("train", "test"):
        f = Path(odometry_dir) / split / f"{seq}.json"
        if f.exists():
            return json.load(open(f))["odometry"]
    raise FileNotFoundError(f"no odometry for {seq} under {odometry_dir}")


def load_camera(raw, camera=CAMERA):
    """Forward camera calibration: K, distortion, T_camera_from_ego (robot/label frame -> optical frame)."""
    cams = yaml.safe_load(open(Path(raw) / "calibration/cameras.yaml"))["cameras"]
    s = cams[f"sensor_{camera.split('_')[1]}"]
    f = lambda v: np.array([float(x) for x in v.split()])  # noqa: E731
    T = np.eye(4)
    T[:3, :3] = f(s["R"]).reshape(3, 3) @ _AXES
    T[:3, 3] = f(s["T"]) / 1000.0
    return dict(name=camera, width=int(s["width"]), height=int(s["height"]), K=f(s["K"]).reshape(3, 3),
                distortion=f(s["D"]), T_camera_from_ego=T)


def lidar_to_ego(points, which):
    """Points of one Velodyne (sensor frame) -> the robot/label frame."""
    p = UPPER_LIDAR if which == "upper" else LOWER_LIDAR
    c, s = np.cos(p["yaw"]), np.sin(p["yaw"])
    return points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + np.array(p["translation"])


def load_segments(raw, odometry_dir, seq):
    """Contiguously timestamped stretches of one sequence, in the layout of hnod.coda.load_segments.

    All tracks are pedestrians.  Adds point_keys / image_paths / camera for the converters.
    """
    labels = json.load(open(Path(raw) / f"labels/labels_3d/{seq}.json"))["labels"]
    odom = load_odometry(odometry_dir, seq)
    ts_all = load_timestamps(raw, seq)
    names = sorted(set(labels) & set(odom) & set(ts_all))
    ts = np.array([ts_all[n] for n in names])
    cuts = np.flatnonzero(np.diff(ts) > MAX_STEP_S) + 1
    camera = load_camera(raw)
    segments = []
    for k, part in enumerate(np.split(np.arange(len(names)), cuts)):
        F = len(part)
        seg_names = [names[i] for i in part]
        T = np.tile(np.eye(4), (F, 1, 1))
        for i, name in enumerate(seg_names):
            p, q = odom[name]["position"], odom[name]["orientation"]
            T[i, :3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
            T[i, :3, 3] = (p["x"], p["y"], p["z"])
        yaw = np.arctan2(T[:, 1, 0], T[:, 0, 0])
        tracks, bottoms = {}, []
        for i, name in enumerate(seg_names):
            for lab in labels[name]:
                b = lab["box"]
                tr = tracks.get(lab["label_id"])
                if tr is None:
                    tr = tracks[lab["label_id"]] = dict(
                        category="Pedestrian", xyz=np.full((F, 3), np.nan), yaw=np.full(F, np.nan),
                        lwh=np.full((F, 3), np.nan), occlusion=np.full(F, "", dtype=object), valid=np.zeros(F, bool))
                tr["xyz"][i] = T[i, :3, :3] @ np.array([b["cx"], b["cy"], b["cz"]]) + T[i, :3, 3]
                tr["yaw"][i] = yaw[i] + b["rot_z"]
                tr["lwh"][i] = (b["l"], b["w"], b["h"])
                tr["occlusion"][i] = "Unknown"
                tr["valid"][i] = True
                if np.hypot(b["cx"], b["cy"]) < 8:
                    bottoms.append(b["cz"] - b["h"] / 2)
        # Nearby pedestrians stand on the floor: their box bottoms give the origin's height above it.
        origin_height = float(-np.median(bottoms)) if bottoms else 0.76
        segments.append(dict(
            dataset="jrdb", sequence=seq, segment=k, rate_hz=RATE_HZ, frames=part, timestamps=ts[part], ego_T=T,
            labelled=np.ones(F, bool), ego_size=EGO_SIZE, lidar_height=origin_height, tracks=tracks,
            camera=camera, point_keys=[(seq, n) for n in seg_names], bev_keys=[(seq, n) for n in seg_names],
            image_paths=[f"{seq}/{Path(n).stem}.jpg" for n in seg_names], frame_names=seg_names))
    return segments
