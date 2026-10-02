"""Compose per-frame BEV grids into an ego-aligned static obstacle map per scenario.

Map image convention (uint8, MAP_SIZE x MAP_SIZE, MAP_RES m/pixel, ROS-like):
    0 = occupied, 255 = observed free ground, 127 = unknown.
The ego at the current step sits at the image centre, +x (forward) points up
and +y (left) points left:
    x = MAP_RANGE - (row + 0.5) * MAP_RES
    y = MAP_RANGE - (col + 0.5) * MAP_RES
"""
import zlib

import cv2
import numpy as np
from scipy import ndimage

from .lidar_bev import FRAME_GRID, FREE, OCCUPIED, RES, UNKNOWN
from .scenario import STATIC

MAP_RES = RES
MAP_SIZE = 400
MAP_RANGE = MAP_SIZE * MAP_RES / 2
CANVAS = 640                 # world-aligned working canvas, large enough for any rotation
BOX_MARGIN = 0.3             # movable boxes are grown by this before being cut out
MIN_OCC_FRAMES = 2
MIN_OCC_RATIO = 0.15
PIX_OCCUPIED, PIX_UNKNOWN, PIX_FREE = 0, 127, 255


def masked_frame_grids(seg, bev):
    """Decode the BEV grid of every segment frame and blank out movable objects.

    bev: {key: ((ix0, iy0), zlib blob)}, keyed by seg["bev_keys"] if present, else by
    source frame number.  Returns {segment_index: (origin, grid)}.
    Movable objects (pedestrians, cycles, vehicles...) are represented by their
    boxes, so their lidar returns must not be baked into the static map.
    """
    movable = np.flatnonzero(seg["types"] != STATIC)
    keys = seg["bev_keys"] if "bev_keys" in seg else [int(f) for f in seg["frames"]]
    out = {}
    for i, key in enumerate(keys):
        if key not in bev:
            continue
        origin, blob = bev[key]
        grid = np.frombuffer(zlib.decompress(blob), np.uint8).reshape(FRAME_GRID, FRAME_GRID).copy()
        for n in movable[seg["valid"][movable, i]]:
            l, w = seg["dims"][n, 0] / 2 + BOX_MARGIN, seg["dims"][n, 1] / 2 + BOX_MARGIN
            c, s = np.cos(seg["yaw"][n, i]), np.sin(seg["yaw"][n, i])
            corners = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) @ np.array([[c, s], [-s, c]])
            cells = (corners + seg["xyz"][n, i, :2]) / RES - np.array(origin)
            # One call per box: a joint fillPoly leaves overlaps between boxes unfilled.
            cv2.fillConvexPoly(grid, np.round(cells[:, ::-1]).astype(np.int32), UNKNOWN)  # (col, row) = (j, i)
        out[i] = (origin, grid)
    return out


def compose_map(grids, contrib, centre_xy, yaw, min_span, close_cells=5):
    """Merge frame grids and resample into the ego-aligned map image.

    grids: output of masked_frame_grids.  contrib: segment indices to merge.
    centre_xy, yaw: ego position/heading (world) at the scenario's current step.
    min_span: a cell only counts as a static obstacle if its first and last
    occupied observations are at least this many frames apart (capped at half
    the merged frames).  Not every moving object is labelled in every frame, so
    the box cut-out alone leaves streaks along their paths; a moving object
    occupies a given cell only briefly, a static one for as long as it is seen.
    close_cells: unknown gaps between lidar rings on open ground up to this many
    cells wide are filled as free (sparser lidars need more).
    """
    c0 = np.floor(np.asarray(centre_xy) / RES).astype(np.int64) - CANVAS // 2
    occ = np.zeros((CANVAS, CANVAS), np.uint16)
    free = np.zeros((CANVAS, CANVAS), np.uint16)
    first = np.full((CANVAS, CANVAS), np.iinfo(np.int32).max, np.int32)
    last = np.full((CANVAS, CANVAS), -1, np.int32)
    used = 0
    for i in contrib:
        if i not in grids:
            continue
        used += 1
        (ox, oy), g = grids[i]
        di, dj = int(ox - c0[0]), int(oy - c0[1])
        a0, a1 = max(di, 0), min(di + FRAME_GRID, CANVAS)
        b0, b1 = max(dj, 0), min(dj + FRAME_GRID, CANVAS)
        if a0 >= a1 or b0 >= b1:
            continue
        sub = g[a0 - di:a1 - di, b0 - dj:b1 - dj]
        hit = sub == OCCUPIED
        occ[a0:a1, b0:b1] += hit
        free[a0:a1, b0:b1] += sub == FREE
        np.minimum(first[a0:a1, b0:b1], i, out=first[a0:a1, b0:b1], where=hit)
        np.maximum(last[a0:a1, b0:b1], i, out=last[a0:a1, b0:b1], where=hit)
    occ_m = (occ >= MIN_OCC_FRAMES) & (occ >= MIN_OCC_RATIO * (occ.astype(np.float32) + free)) \
        & (last - first >= min(min_span, used // 2))
    free_m = (free > 0) & ~occ_m
    # Lidar rings leave thin unobserved gaps on open ground; close them.
    free_m = ndimage.binary_closing(free_m | occ_m, structure=np.ones((close_cells, close_cells), bool)) & ~occ_m

    rc = MAP_RANGE - (np.arange(MAP_SIZE, dtype=np.float32) + 0.5) * MAP_RES
    xs, ys = np.meshgrid(rc, rc, indexing="ij")  # xs varies with row, ys with column
    c, s = np.cos(yaw), np.sin(yaw)
    ii = ((centre_xy[0] + c * xs - s * ys) / RES - c0[0] - 0.5).astype(np.float32)
    jj = ((centre_xy[1] + s * xs + c * ys) / RES - c0[1] - 0.5).astype(np.float32)
    occ_r = cv2.remap(occ_m.astype(np.float32), jj, ii, cv2.INTER_LINEAR, borderValue=0) >= 0.3
    free_r = (cv2.remap(free_m.astype(np.float32), jj, ii, cv2.INTER_LINEAR, borderValue=0) >= 0.5) & ~occ_r
    img = np.full((MAP_SIZE, MAP_SIZE), PIX_UNKNOWN, np.uint8)
    img[free_r] = PIX_FREE
    img[occ_r] = PIX_OCCUPIED
    return img


def xy_to_rowcol(x, y):
    """Scenario-frame metres -> fractional (row, col) in the map image."""
    return (MAP_RANGE - np.asarray(x)) / MAP_RES - 0.5, (MAP_RANGE - np.asarray(y)) / MAP_RES - 0.5
