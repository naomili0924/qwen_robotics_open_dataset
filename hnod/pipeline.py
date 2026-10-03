"""Shared last stage of every converter: cleaned segment -> scenario rows."""
from pathlib import Path

import numpy as np
from datasets import Dataset

from . import maps, scenario
from .io import to_row
from .points import scenario_points


def segment_rows(seg, bev, step, stride, map_context_frames, min_static_span, close_cells=5,
                 points=None, camera=None, min_future_distance=0.0):
    """Rows for all scenarios of one cleaned segment (see scenario.clean_segment).

    bev: per-frame lidar grids as accepted by maps.masked_frame_grids ({} for none).
    map_context_frames: static_map merges sweeps this many frames either side of the
    current step.  min_static_span: frames an obstacle cell must persist to count as
    static.  close_cells: see maps.compose_map.
    points: per-frame point store for points.scenario_points (needs seg["point_keys"]).
    camera: callable(seg, sc) -> (image bytes for steps 0..current, camera dict), using
    seg["image_paths"].  A scenario is skipped if a requested image or sweep is missing,
    or if the robot covers less than min_future_distance metres over the future steps.
    """
    grids = maps.masked_frame_grids(seg, bev) if bev else {}
    F = len(seg["frames"])
    rows = []
    for a in scenario.window_anchors(seg, step, stride):
        sc = scenario.build_scenario(seg, a, step)
        future = np.stack([sc["ego"]["x"], sc["ego"]["y"]], 1)[scenario.CURRENT:]
        if np.linalg.norm(np.diff(future, axis=0), axis=1).sum() < min_future_distance:
            continue
        images = cam = lidar = static_map = None
        if camera is not None:
            images, cam = camera(seg, sc)
            if images is None:
                continue
        if points is not None:
            lidar = scenario_points(seg, sc, points)
            if lidar is None:
                continue
        if grids:
            around = range(max(0, a - map_context_frames), min(F, a + map_context_frames + 1))
            static_map = maps.compose_map(grids, around, seg["ego_xyz"][a, :2], seg["ego_yaw"][a], min_static_span,
                                          close_cells)
        rows.append(to_row(sc, static_map, maps.MAP_RES, images, cam, lidar))
    return rows


def camera_inputs(seg, sc):
    """Past-and-current images and camera geometry for one scenario.

    Needs seg["image_paths"] (one path or None per segment frame) and seg["camera"]
    (dict(name, width, height, K, T_camera_from_ego)).  Returns (None, None) if an
    image is missing.  T_scenario_from_camera has one camera pose per image.
    """
    idx = sc["segment_index"][:sc["current_time_index"] + 1]
    paths = [seg["image_paths"][i] for i in idx]
    if any(p is None or not Path(p).exists() for p in paths):
        return None, None
    cam = seg["camera"]
    ego_from_camera = np.linalg.inv(cam["T_camera_from_ego"])
    scenario_from_world = np.linalg.inv(sc["world_from_scenario"])
    poses = [(scenario_from_world @ seg["ego_T"][i] @ ego_from_camera).reshape(-1).tolist() for i in idx]
    return [Path(p).read_bytes() for p in paths], dict(
        name=cam["name"], width=cam["width"], height=cam["height"], K=cam["K"].reshape(-1).tolist(),
        T_scenario_from_camera=poses)


def write_shards(rows, out_dir, split, features, shard_rows=500):
    """Write rows as <split>-NNNNN-of-NNNNN.parquet under out_dir."""
    if not rows:
        return 0
    rows.sort(key=lambda r: r["scenario_id"])
    n = int(np.ceil(len(rows) / shard_rows))
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        Dataset.from_list(rows[k * shard_rows:(k + 1) * shard_rows], features=features) \
            .to_parquet(d / f"{split}-{k:05d}-of-{n:05d}.parquet")
    return n
