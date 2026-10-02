"""Shared last stage of every converter: cleaned segment -> scenario rows."""
from pathlib import Path

import numpy as np
from datasets import Dataset

from . import maps, scenario
from .io import to_row


def segment_rows(seg, bev, step, stride, map_context_frames, min_static_span, close_cells=5):
    """Rows for all scenarios of one cleaned segment (see scenario.clean_segment).

    bev: per-frame lidar grids as accepted by maps.masked_frame_grids ({} for none).
    map_context_frames: static_map merges sweeps this many frames either side of the
    current step, observed_map this many before it.  min_static_span: frames an
    obstacle cell must persist to count as static.  close_cells: see maps.compose_map.
    """
    grids = maps.masked_frame_grids(seg, bev) if bev else {}
    F = len(seg["frames"])
    rows = []
    for a in scenario.window_anchors(seg, step, stride):
        sc = scenario.build_scenario(seg, a, step)
        static_map = observed_map = None
        if grids:
            centre, yaw = seg["ego_xyz"][a, :2], seg["ego_yaw"][a]
            around = range(max(0, a - map_context_frames), min(F, a + map_context_frames + 1))
            past = range(max(0, a - map_context_frames), a + 1)
            static_map = maps.compose_map(grids, around, centre, yaw, min_static_span, close_cells)
            observed_map = maps.compose_map(grids, past, centre, yaw, min_static_span, close_cells)
        rows.append(to_row(sc, static_map, observed_map, maps.MAP_RES))
    return rows


def write_shards(rows, out_dir, split, features, shard_rows=500):
    """Write rows as <split>-NNNNN-of-NNNNN.parquet under out_dir."""
    rows.sort(key=lambda r: r["scenario_id"])
    n = int(np.ceil(len(rows) / shard_rows))
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        Dataset.from_list(rows[k * shard_rows:(k + 1) * shard_rows], features=features) \
            .to_parquet(d / f"{split}-{k:05d}-of-{n:05d}.parquet")
    return n
