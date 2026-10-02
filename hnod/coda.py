"""Reader for the annotation-level parts of CODa (UT Campus Object Dataset)."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Height of the OS1 lidar origin above the ground the robot stands on [m].
# Estimated from the bottom faces of nearby annotated pedestrians (median 0.79 m).
LIDAR_HEIGHT = 0.8
# Clearpath Husky footprint (length, width) and height including the sensor mast [m].
EGO_SIZE = (0.99, 0.67, 1.0)
RATE_HZ = 10.0

_FRAME_RE = re.compile(r"_(\d+)\.json$")


def load_poses(raw, seq):
    """Lidar -> world transforms for every frame of a sequence, shape (F, 4, 4).

    Globally optimised poses are used where CODa provides them; sequences
    8, 14 and 15 only have odometry-quality poses (locally consistent).
    """
    raw = Path(raw)
    f = raw / f"poses/dense_global/{seq}.txt"
    if not f.exists():
        f = raw / f"poses/dense/{seq}.txt"
    p = np.loadtxt(f)
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(p[:, [5, 6, 7, 4]]).as_matrix()  # file is qw qx qy qz
    T[:, :3, 3] = p[:, 1:4]
    return T


def load_timestamps(raw, seq):
    return np.loadtxt(Path(raw) / f"timestamps/{seq}.txt")


def annotated_frames(raw, seq):
    d = Path(raw) / f"3d_bbox/os1/{seq}"
    return sorted(int(_FRAME_RE.search(p.name).group(1)) for p in d.glob("*.json"))


def load_boxes(raw, seq, frame):
    with open(Path(raw) / f"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{frame}.json") as f:
        return json.load(f)["3dbbox"]


def sequences(raw):
    return sorted(int(p.name) for p in (Path(raw) / "3d_bbox/os1").iterdir())


def _yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def load_segments(raw, seq, max_gap=2):
    """Split a sequence into contiguously annotated segments.

    CODa labels come in bursts of consecutive 10 Hz frames; instance ids are only
    consistent inside a burst.  A single dropped label frame (gap of 2) is kept
    inside the segment and filled later by interpolation.

    Returns a list of dicts with world-frame arrays indexed by segment frame:
      frames (F,), timestamps (F,), ego_T (F,4,4), labelled (F,) bool,
      tracks: {instance_id: dict(category, xyz (F,3), yaw (F,), lwh (F,3),
                                  occlusion (F,) object, valid (F,) bool)}
    """
    frames = np.array(annotated_frames(raw, seq))
    poses = load_poses(raw, seq)
    ts = load_timestamps(raw, seq)
    cuts = np.flatnonzero(np.diff(frames) > max_gap) + 1
    segments = []
    for k, run in enumerate(np.split(frames, cuts)):
        f0, f1 = int(run[0]), int(run[-1])
        F = f1 - f0 + 1
        seg = dict(dataset="coda", sequence=str(seq), segment=k, rate_hz=RATE_HZ,
                   frames=np.arange(f0, f1 + 1), timestamps=ts[f0:f1 + 1], ego_T=poses[f0:f1 + 1],
                   labelled=np.zeros(F, bool), ego_size=EGO_SIZE, lidar_height=LIDAR_HEIGHT, tracks={})
        for frame in run:
            i = int(frame) - f0
            seg["labelled"][i] = True
            T = seg["ego_T"][i]
            ego_yaw = _yaw_of(T[:3, :3])
            for b in load_boxes(raw, seq, int(frame)):
                tr = seg["tracks"].get(b["instanceId"])
                if tr is None:
                    tr = seg["tracks"][b["instanceId"]] = dict(
                        category=b["classId"], xyz=np.full((F, 3), np.nan), yaw=np.full(F, np.nan),
                        lwh=np.full((F, 3), np.nan), occlusion=np.full(F, "", dtype=object), valid=np.zeros(F, bool))
                tr["xyz"][i] = T[:3, :3] @ np.array([b["cX"], b["cY"], b["cZ"]]) + T[:3, 3]
                tr["yaw"][i] = ego_yaw + b["y"]
                tr["lwh"][i] = (b["l"], b["w"], b["h"])
                tr["occlusion"][i] = str(b["labelAttributes"].get("isOccluded", "Unknown")).capitalize()
                tr["valid"][i] = True
        segments.append(seg)
    return segments
