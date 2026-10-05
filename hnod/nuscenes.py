"""Reader for nuScenes (v1.0-trainval / v1.0-mini), straight from its JSON tables (no devkit).

Layout under the data root:
  v1.0-*/{scene,sample,sample_data,ego_pose,calibrated_sensor,sensor,sample_annotation,instance,
          category,visibility}.json
  samples/CAM_FRONT/*.jpg, samples/LIDAR_TOP/*.pcd.bin      (key frames, 2 Hz)

Conventions (nuScenes documentation):
  * the ego frame has its origin at the midpoint of the rear axle projected onto the ground,
    x forward, y left, z up; ego_pose is ego -> global
  * calibrated_sensor is sensor -> ego; camera frames are optical (x right, y down, z forward)
  * rotations are quaternions (w, x, y, z); box size is (width, length, height); box
    translation is the box centre in the global frame
  * LIDAR_TOP files are float32 (N, 5): x, y, z, intensity, ring, in the lidar frame

Only key frames carry annotations, so segments are at 2 Hz.  NOT YET RUN ON REAL DATA: written
against the published format and exercised only with a synthetic fixture (tests/test_nuscenes.py).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .scenario import CYCLE, OTHER_MOVABLE, PEDESTRIAN, STATIC, VEHICLE

RATE_HZ = 2.0
MAX_STEP_S = 0.75        # key frames further apart than this start a new segment
EGO_SIZE = (4.08, 1.73, 1.56)   # Renault Zoe
CAMERA, LIDAR = "CAM_FRONT", "LIDAR_TOP"


def type_of(category):
    """nuScenes category name -> coarse type."""
    if category.startswith("human.pedestrian"):
        return PEDESTRIAN
    if category in ("vehicle.bicycle", "vehicle.motorcycle"):
        return CYCLE
    if category.startswith("vehicle."):
        return VEHICLE
    if category in ("movable_object.barrier", "movable_object.trafficcone") or category.startswith("static_object"):
        return STATIC
    return OTHER_MOVABLE     # animals, pushable / pullable objects, debris


def _T(rotation_wxyz, translation):
    w, x, y, z = rotation_wxyz
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = translation
    return T


class Tables:
    """The JSON tables of one nuScenes version, indexed by token."""

    def __init__(self, root, version="v1.0-trainval"):
        self.root = Path(root)
        d = self.root / version
        load = lambda name: json.load(open(d / f"{name}.json"))  # noqa: E731
        by_token = lambda rows: {r["token"]: r for r in rows}  # noqa: E731
        self.scenes = load("scene")
        self.sample = by_token(load("sample"))
        self.ego_pose = by_token(load("ego_pose"))
        self.calibrated = by_token(load("calibrated_sensor"))
        sensor = by_token(load("sensor"))
        category = by_token(load("category"))
        self.instance_category = {r["token"]: category[r["category_token"]]["name"] for r in load("instance")}
        vis = d / "visibility.json"
        self.visibility = {r["token"]: r.get("level", "") for r in json.load(open(vis))} if vis.exists() else {}
        # key-frame sample_data of the two channels used, by sample token
        self.data = {CAMERA: {}, LIDAR: {}}
        for r in load("sample_data"):
            if not r["is_key_frame"]:
                continue
            channel = sensor[self.calibrated[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            if channel in self.data:
                self.data[channel][r["sample_token"]] = r
        self.annotations = {}
        for r in load("sample_annotation"):
            self.annotations.setdefault(r["sample_token"], []).append(r)

    def scene_names(self):
        return sorted(s["name"] for s in self.scenes)

    def samples_of(self, scene_name):
        scene = next(s for s in self.scenes if s["name"] == scene_name)
        out, tok = [], scene["first_sample_token"]
        while tok:
            out.append(self.sample[tok])
            tok = self.sample[tok]["next"]
        return out


def load_segments(tables, scene_name):
    """Contiguous key-frame stretches of one scene, in the layout of hnod.coda.load_segments.

    Adds camera, image_paths, lidar_paths, lidar_T (lidar -> world per frame), lidar_heights,
    point_keys / bev_keys for the converter.
    """
    samples = [s for s in tables.samples_of(scene_name)
               if s["token"] in tables.data[LIDAR] and s["token"] in tables.data[CAMERA]]
    if not samples:
        return []
    ts = np.array([s["timestamp"] for s in samples]) / 1e6
    cuts = np.flatnonzero(np.diff(ts) > MAX_STEP_S) + 1
    segments = []
    for k, part in enumerate(np.split(np.arange(len(samples)), cuts)):
        F = len(part)
        ego_T, lidar_T = np.zeros((F, 4, 4)), np.zeros((F, 4, 4))
        lidar_heights, images, sweeps, tracks = np.zeros(F), [], [], {}
        camera = None
        for i, j in enumerate(part):
            s = samples[j]
            lid, cam = tables.data[LIDAR][s["token"]], tables.data[CAMERA][s["token"]]
            pose = tables.ego_pose[lid["ego_pose_token"]]
            ego_T[i] = _T(pose["rotation"], pose["translation"])
            cal = tables.calibrated[lid["calibrated_sensor_token"]]
            lidar_T[i] = ego_T[i] @ _T(cal["rotation"], cal["translation"])
            lidar_heights[i] = cal["translation"][2]
            images.append(str(tables.root / cam["filename"]))
            sweeps.append(str(tables.root / lid["filename"]))
            if camera is None:
                cc = tables.calibrated[cam["calibrated_sensor_token"]]
                camera = dict(name=CAMERA, width=int(cam.get("width") or 1600), height=int(cam.get("height") or 900),
                              K=np.array(cc["camera_intrinsic"], dtype=np.float64),
                              T_camera_from_ego=np.linalg.inv(_T(cc["rotation"], cc["translation"])))
            for a in tables.annotations.get(s["token"], []):
                tr = tracks.get(a["instance_token"])
                if tr is None:
                    tr = tracks[a["instance_token"]] = dict(
                        category=tables.instance_category[a["instance_token"]], xyz=np.full((F, 3), np.nan),
                        yaw=np.full(F, np.nan), lwh=np.full((F, 3), np.nan), occlusion=np.full(F, "", dtype=object),
                        valid=np.zeros(F, bool))
                w, l, h = a["size"]
                R = _T(a["rotation"], (0, 0, 0))[:3, :3]
                tr["xyz"][i] = a["translation"]
                tr["yaw"][i] = np.arctan2(R[1, 0], R[0, 0])
                tr["lwh"][i] = (l, w, h)
                tr["occlusion"][i] = "Unknown"
                tr["valid"][i] = True
        keys = [(scene_name, int(j)) for j in part]
        segments.append(dict(
            dataset="nuscenes", sequence=scene_name, segment=k, rate_hz=RATE_HZ, frames=part, timestamps=ts[part],
            ego_T=ego_T, labelled=np.ones(F, bool), ego_size=EGO_SIZE, lidar_height=0.0, tracks=tracks,
            camera=camera, image_paths=images, lidar_paths=sweeps, lidar_T=lidar_T, lidar_heights=lidar_heights,
            point_keys=keys, bev_keys=keys))
    return segments


def load_sweep(path):
    """LIDAR_TOP .pcd.bin -> (N, 3) float64 points in the lidar frame."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
