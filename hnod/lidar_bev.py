"""Per-frame lidar -> bird's-eye-view traversability grid.

Each lidar sweep is turned into a small world-axis-aligned tri-state grid
(unknown / free ground / obstacle) centred on the sensor.  Because every frame
grid is aligned to the same world axes and snapped to the same 0.1 m lattice,
grids from different frames can be merged with pure integer shifts.
"""
import zlib

import numpy as np
from scipy import ndimage

RES = 0.1            # fine grid resolution [m]
FRAME_GRID = 512     # per-frame grid is FRAME_GRID x FRAME_GRID cells (51.2 m)
COARSE = 8           # coarse ground cell = COARSE fine cells (0.8 m)
GROUND_CELLS = 96    # coarse ground grid is GROUND_CELLS x GROUND_CELLS, centred on the sensor
GROUND_BAND = 0.35   # ground may differ this much from the height expected from inner neighbours [m]
SLACK_PER_CELL = 0.15  # band growth per coarse cell crossed without seeing ground [m]
MAX_SLACK = 0.6
MIN_GROUND_PTS = 3
MIN_RANGE = 0.9      # drop returns from the robot itself
OBST_MIN_H = 0.2     # points this far above ground count as obstacles ...
OBST_MAX_H = 2.0     # ... unless they are above head height
GROUND_MAX_H = 0.1   # points this close to the ground mark the cell as free
MIN_OBST_PTS = 2     # obstacle points needed in a cell

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def estimate_ground(rel, lidar_height):
    """Ground height (relative to the sensor) on the coarse grid.

    rel: (N, 3) points in the sensor frame.  Working in the sensor frame (which
    tilts with the robot, so the ground under it is always near z = -lidar_height)
    makes the result independent of any tilt or drift in the pose's world frame.
    The coarse grid spans +-GROUND_CELLS/2 cells so it covers the whole
    world-aligned frame grid whatever the robot's yaw.

    The ground is grown outwards from the robot, ring by ring.  A cell takes the
    low percentile of its points that lie within a band around the height
    expected from its already-solved inner neighbours; a cell with no such
    points inherits the expected height.  Growing from the known ground under
    the robot (rather than trusting the lowest return in each cell) keeps
    below-ground ghost returns - common on wet, reflective pavement - from
    dragging the surface down.
    """
    n = GROUND_CELLS
    step = COARSE * RES
    half = n * step / 2
    ci = np.clip(((rel[:, 0] + half) / step).astype(np.int32), 0, n - 1)
    cj = np.clip(((rel[:, 1] + half) / step).astype(np.int32), 0, n - 1)
    ring = np.maximum(np.abs(ci - (n - 1) / 2), np.abs(cj - (n - 1) / 2)).astype(np.int32)
    cell = ci * n + cj
    order = np.lexsort((rel[:, 2], cell, ring))
    ring_s, cell_s, z_s = ring[order], cell[order], rel[order, 2]
    ring_start = np.searchsorted(ring_s, np.arange(n // 2 + 1))

    ii, jj = np.indices((n, n))
    cell_ring = np.maximum(np.abs(ii - (n - 1) / 2), np.abs(jj - (n - 1) / 2)).astype(np.int32)
    g = np.full((n + 2, n + 2), np.nan, dtype=np.float32)    # padded by one cell
    slack = np.full((n + 2, n + 2), np.nan, dtype=np.float32)  # extra band width where ground was not seen
    centre = cell_ring == 0
    g[1:-1, 1:-1][centre] = -lidar_height  # the robot stands on the ground
    slack[1:-1, 1:-1][centre] = 0.0
    for r in range(1, n // 2):
        ri, rj = np.nonzero(cell_ring == r)
        # expected height / slack from the solved neighbours (all in ring r-1)
        nb_g = np.stack([g[ri + 1 + di, rj + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        nb_s = np.stack([slack[ri + 1 + di, rj + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        ref, ref_slack = np.nanmean(nb_g, axis=0), np.nanmin(nb_s, axis=0)
        ref_grid = np.full(n * n, np.nan, dtype=np.float32)
        band_grid = np.zeros(n * n, dtype=np.float32)
        ref_grid[ri * n + rj] = ref
        band_grid[ri * n + rj] = GROUND_BAND + ref_slack
        new_g, new_slack = ref.copy(), np.minimum(ref_slack + SLACK_PER_CELL, MAX_SLACK)

        sl = slice(ring_start[r], ring_start[r + 1])
        c, z = cell_s[sl], z_s[sl]
        ok = np.abs(z - ref_grid[c]) <= band_grid[c]
        c, z = c[ok], z[ok]
        if c.size:
            starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            counts = np.diff(np.r_[starts, c.size])
            good = counts >= MIN_GROUND_PTS
            pick = starts + (counts * 0.1).astype(np.int64)
            lut = np.full(n * n, -1, dtype=np.int64)
            lut[ri * n + rj] = np.arange(ri.size)
            k = lut[c[starts][good]]
            new_g[k] = z[pick[good]]
            new_slack[k] = 0.0
        g[ri + 1, rj + 1] = new_g
        slack[ri + 1, rj + 1] = new_slack
    return g[1:-1, 1:-1]


def point_heights(points, R, t, lidar_height):
    """Height above the estimated ground for every point that falls in the grid.

    points: (N, 3+) lidar-frame points.  R, t: lidar -> world rotation/translation.
    Returns (idx, fi, fj, h, origin): indices into `points`, fine-grid cell of
    each kept point, its height above ground, and the grid origin (ix0, iy0).
    """
    p = points[:, :3]
    rng = np.linalg.norm(p, axis=1)
    idx = np.flatnonzero((rng > MIN_RANGE) & np.isfinite(rng))
    pw = p[idx] @ R.T + t
    origin = np.floor(t[:2] / RES).astype(np.int64) - FRAME_GRID // 2
    fi = np.floor(pw[:, 0] / RES).astype(np.int64) - origin[0]
    fj = np.floor(pw[:, 1] / RES).astype(np.int64) - origin[1]
    keep = (fi >= 0) & (fi < FRAME_GRID) & (fj >= 0) & (fj < FRAME_GRID)
    idx, fi, fj = idx[keep], fi[keep], fj[keep]
    rel = p[idx].astype(np.float64)
    ground = estimate_ground(rel, lidar_height)
    step = COARSE * RES
    half = GROUND_CELLS * step / 2
    coords = np.stack([(rel[:, 0] + half) / step - 0.5, (rel[:, 1] + half) / step - 0.5])
    h = rel[:, 2] - ndimage.map_coordinates(ground, coords, order=1, mode="nearest")
    return idx, fi, fj, h, (int(origin[0]), int(origin[1]))


def rasterize(fi, fj, h):
    """Tri-state grid from per-point fine cells and heights above ground."""
    flat = fi * FRAME_GRID + fj
    size = FRAME_GRID * FRAME_GRID
    obst = np.bincount(flat[(h > OBST_MIN_H) & (h < OBST_MAX_H)], minlength=size)
    free = np.bincount(flat[h < GROUND_MAX_H], minlength=size)
    grid = np.zeros(size, dtype=np.uint8)
    grid[free > 0] = FREE
    grid[obst >= MIN_OBST_PTS] = OCCUPIED
    return grid.reshape(FRAME_GRID, FRAME_GRID)


def frame_grid(points, R, t, lidar_height):
    """Tri-state BEV grid for one sweep.

    Returns (grid uint8 [FRAME_GRID, FRAME_GRID], origin (ix0, iy0)) where
    grid[i, j] covers world x in [(ix0+i)*RES, (ix0+i+1)*RES) and likewise for y.
    """
    _, fi, fj, h, origin = point_heights(points, R, t, lidar_height)
    return rasterize(fi, fj, h), origin


POINT_RANGE = 28.0        # keep points within this horizontal range of the sensor [m]
POINT_MAX_H = 3.0         # ... and no higher than this above the ground
POINT_GROUND_H = 0.15     # points lower than this above the ground are flagged as ground
OBSTACLE_VOXEL = 0.1      # one point is kept per voxel of this size for non-ground points [m] ...
GROUND_VOXEL = 0.25       # ... and per horizontal cell of this size for ground points


def downsample_points(xyz, h):
    """Thin a sweep to the points worth publishing.

    xyz: (N, 3) sensor-frame points, h: their height above the estimated ground
    (as returned by point_heights for the same points).  Returns (xyz float32 (M, 3),
    is_ground bool (M,)).
    """
    keep = (np.linalg.norm(xyz[:, :2], axis=1) <= POINT_RANGE) & (h <= POINT_MAX_H)
    xyz, h = xyz[keep], h[keep]
    ground = h < POINT_GROUND_H
    out = []
    for mask, voxel, dims in ((~ground, OBSTACLE_VOXEL, 3), (ground, GROUND_VOXEL, 2)):
        p = xyz[mask]
        cell = np.floor(p[:, :dims] / voxel).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        out.append(p[np.sort(first)])
    return (np.concatenate(out).astype(np.float32),
            np.r_[np.zeros(len(out[0]), bool), np.ones(len(out[1]), bool)])


def pack_points(xyz, is_ground):
    """Compress downsample_points output: centimetre int16 coordinates + packed flags."""
    return (zlib.compress(np.round(xyz * 100).astype(np.int16).tobytes(), 3),
            zlib.compress(np.packbits(is_ground).tobytes(), 3), len(xyz))


def unpack_points(packed):
    """Inverse of pack_points: (xyz float32 metres (M, 3), is_ground bool (M,))."""
    xyz_blob, ground_blob, n = packed
    xyz = np.frombuffer(zlib.decompress(xyz_blob), np.int16).reshape(-1, 3).astype(np.float32) / 100
    return xyz, np.unpackbits(np.frombuffer(zlib.decompress(ground_blob), np.uint8))[:n].astype(bool)
