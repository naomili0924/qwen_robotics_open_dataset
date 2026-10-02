"""Per-step lidar point clouds for the current and future steps of a scenario.

Each published point is in the scenario frame (centimetres, int16) and carries
  label  0 = ground, 1 = static obstacle, 2 = dynamic object
  track  index into the scenario's `future_tracks` of the box the point lies in, or -1
A point inside a labelled box is dynamic if that track moves during the scenario
and static otherwise; unlabelled structure (walls, kerbs, vegetation) is static.
"""
import numpy as np

from .lidar_bev import unpack_points
from .maps import MAP_RANGE
from .scenario import CURRENT

LABEL_GROUND, LABEL_STATIC, LABEL_DYNAMIC = 0, 1, 2
BOX_MARGIN_XY = 0.1   # boxes are grown by this when collecting their points [m]
BOX_MARGIN_Z = 0.15


def scenario_points(seg, sc, store):
    """Point clouds for steps CURRENT.. of scenario `sc`, or None if a sweep is missing.

    store: {key: packed points} with keys seg["point_keys"][i] (sensor-frame points
    from lidar_bev.pack_points).  seg["lidar_T"] (sensor -> world per frame) defaults
    to the ego pose.  Returns dict(x, y, z, label, track), each a list with one
    array per step.
    """
    scenario_from_world = np.linalg.inv(sc["world_from_scenario"])
    sensor_T = seg.get("lidar_T", seg["ego_T"])
    tr = sc["tracks"]
    half = np.stack([tr["length"] / 2 + BOX_MARGIN_XY, tr["width"] / 2 + BOX_MARGIN_XY,
                     tr["height"] / 2 + BOX_MARGIN_Z], axis=1) if len(tr["id"]) else np.zeros((0, 3))
    out = dict(x=[], y=[], z=[], label=[], track=[])
    for step, i in enumerate(sc["segment_index"][CURRENT:], start=CURRENT):
        packed = store.get(seg["point_keys"][i])
        if packed is None:
            return None
        xyz, ground = unpack_points(packed)
        T = scenario_from_world @ sensor_T[i]
        p = xyz @ T[:3, :3].T + T[:3, 3]
        keep = (np.abs(p[:, 0]) < MAP_RANGE) & (np.abs(p[:, 1]) < MAP_RANGE)
        p, ground = p[keep], ground[keep]
        label = np.where(ground, LABEL_GROUND, LABEL_STATIC).astype(np.uint8)
        track = np.full(len(p), -1, np.int16)
        for n in np.flatnonzero(tr["valid"][:, step]):
            d = p - np.array([tr["x"][n, step], tr["y"][n, step], tr["z"][n, step]])
            c, s = np.cos(tr["heading"][n, step]), np.sin(tr["heading"][n, step])
            inside = (np.abs(c * d[:, 0] + s * d[:, 1]) <= half[n, 0]) & (np.abs(-s * d[:, 0] + c * d[:, 1]) <= half[n, 1]) \
                & (np.abs(d[:, 2]) <= half[n, 2]) & ~ground
            track[inside] = n
            if not tr["is_stationary"][n]:
                label[inside] = LABEL_DYNAMIC
        cm = np.round(p * 100).astype(np.int16)
        for k, v in zip(("x", "y", "z", "label", "track"), (cm[:, 0], cm[:, 1], cm[:, 2], label, track)):
            out[k].append(v)
    return out
