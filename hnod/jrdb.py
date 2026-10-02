"""Reader for JRDB (JackRabbot Dataset) 3D pedestrian labels + robot odometry.

STATUS: written against the published label format but NOT yet run on real JRDB
labels (the download needs a registered account).  Expect to adjust details -
in particular the sign convention of `rot_z` - once real files are at hand.

Inputs:
  labels   <labels_dir>/<sequence>.json     JRDB train `labels/labels_3d`
           {"labels": {"000000.pcd": [{"label_id": "pedestrian:12",
                                         "box": {"cx","cy","cz","l","w","h","rot_z"}, ...}]}}
           boxes are in the robot frame: x forward, y left, z up
  odometry <odometry_dir>/<sequence>.json   robot pose per point-cloud frame
           {"odometry": {"000000.pcd": {"position": {x,y,z}, "orientation": {x,y,z,w}}}}
           JRDB ships odometry only inside its rosbags; a pre-extracted copy is public at
           https://storage.googleapis.com/gresearch/human_scene_transformer/odometry.zip
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RATE_HZ = 15.0
# JackRabbot (Segway RMP210 base) approximate footprint and height [m].
EGO_SIZE = (0.65, 0.55, 1.2)


def sequences(labels_dir):
    return sorted(p.stem for p in Path(labels_dir).glob("*.json"))


def load_segment(labels_dir, odometry_dir, seq):
    """One segment per sequence (JRDB sequences are labelled without gaps).

    Same layout as hnod.coda.load_segments; all tracks are pedestrians.
    """
    with open(Path(labels_dir) / f"{seq}.json") as f:
        labels = json.load(f)["labels"]
    with open(Path(odometry_dir) / f"{seq}.json") as f:
        odom = json.load(f)["odometry"]
    names = sorted(set(labels) & set(odom))
    F = len(names)
    T = np.tile(np.eye(4), (F, 1, 1))
    for i, name in enumerate(names):
        p, q = odom[name]["position"], odom[name]["orientation"]
        T[i, :3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        T[i, :3, 3] = (p["x"], p["y"], p["z"])
    yaw = np.arctan2(T[:, 1, 0], T[:, 0, 0])

    tracks, bottoms = {}, []
    for i, name in enumerate(names):
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
    # The label frame's height above the floor is not documented; nearby
    # pedestrians stand on the floor, so their box bottoms give it.
    sensor_height = float(-np.median(bottoms)) if bottoms else 0.0
    return dict(dataset="jrdb", sequence=seq, segment=0, rate_hz=RATE_HZ, frames=np.arange(F),
                timestamps=np.arange(F) / RATE_HZ, ego_T=T, labelled=np.ones(F, bool), ego_size=EGO_SIZE,
                lidar_height=sensor_height, tracks=tracks)
