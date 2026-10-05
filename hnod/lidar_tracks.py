"""Moving people from lidar alone, for sources without object labels (MuSoHu, SCAND).

Derived, not annotated: the tracks stand in for labels so that the evaluator can treat moving people
like any other dataset's (at-fault collisions, people tags).  Per sweep:

1. obstacle points (0.3-2.0 m above the estimated ground) are binned on a world-fixed 0.2 m grid;
2. a point is *static* if its cell is occupied in at least STATIC_FRAMES sweeps within +-WINDOW
   sweeps (walls, furniture, people standing for seconds), otherwise *moving*;
3. moving points are clustered by connected components on a 0.15 m grid (one cell of dilation), and
   person-sized clusters are kept;
4. clusters are linked over time by a constant-velocity gate; tracks seen in at least MIN_HITS
   sweeps that travel at least MIN_TRAVEL are kept as pedestrians.

Standing people end up in the static map instead, which is where a planner must avoid them anyway.
"""
import numpy as np
from scipy import ndimage

STATIC_CELL = 0.2
WINDOW = 50            # sweeps either side (+-5 s at 10 Hz)
STATIC_FRAMES = 20     # occupied in this many sweeps of the window -> static (2 s)
OBST_MIN_H, OBST_MAX_H = 0.3, 2.0
MAX_RANGE = 12.0       # a VLP-16 resolves people only within about this range
CLUSTER_CELL = 0.15
MIN_POINTS = 8
MAX_EXTENT = 1.2       # person-sized clusters only [m]
MIN_HEIGHT_SPAN = 0.3
GATE = 0.8             # association distance after constant-velocity prediction [m]
MAX_MISSES = 3
MIN_HITS = 8           # 0.8 s
MIN_TRAVEL = 0.6       # m
PERSON_LWH = (0.6, 0.6, 1.7)


def moving_points(frames):
    """frames: list of (xy world (N, 2), h (N,) height above ground, sensor_xy (2,)) per sweep.

    Returns per sweep the boolean mask of moving obstacle points (False for non-obstacle points).
    """
    F = len(frames)
    cells, obst = [], []
    for xy, h, sxy in frames:
        m = (h > OBST_MIN_H) & (h < OBST_MAX_H) & (np.hypot(*(xy - sxy).T) < MAX_RANGE)
        obst.append(m)
        c = np.floor(xy[m] / STATIC_CELL).astype(np.int64)
        cells.append(np.unique(c[:, 0] * 1_000_003 + c[:, 1]))  # sorted cell keys of this sweep
    if not F:
        return []
    key = np.concatenate(cells)
    frame = np.concatenate([np.full(len(c), i) for i, c in enumerate(cells)])
    _, rank = np.unique(key, return_inverse=True)
    big = F + 1
    combo = np.sort(rank * big + frame)
    lo = np.searchsorted(combo, rank * big + np.maximum(frame - WINDOW, 0), "left")
    hi = np.searchsorted(combo, rank * big + np.minimum(frame + WINDOW, F - 1), "right")
    count = hi - lo  # sweeps in the window that occupy this sweep's cell
    starts = np.r_[0, np.cumsum([len(c) for c in cells])]
    out = []
    for i, (xy, h, sxy) in enumerate(frames):
        mask = np.zeros(len(xy), bool)
        idx = np.flatnonzero(obst[i])
        if len(idx):
            c = np.floor(xy[idx] / STATIC_CELL).astype(np.int64)
            k = c[:, 0] * 1_000_003 + c[:, 1]
            pos = np.searchsorted(cells[i], k)
            mask[idx[count[starts[i] + pos] < STATIC_FRAMES]] = True
        out.append(mask)
    return out


def clusters(xy, h):
    """Person-sized clusters of moving points: list of (centroid xy, n points)."""
    if len(xy) < MIN_POINTS:
        return []
    c = np.floor(xy / CLUSTER_CELL).astype(np.int64)
    c0 = c.min(0) - 2
    shape = c.max(0) - c0 + 3
    if shape.prod() > 4e6:
        return []
    grid = np.zeros(shape, bool)
    grid[tuple((c - c0).T)] = True
    lab, n = ndimage.label(ndimage.binary_dilation(grid), structure=np.ones((3, 3)))
    pl = lab[tuple((c - c0).T)]
    out = []
    for k in range(1, n + 1):
        sel = pl == k
        if sel.sum() < MIN_POINTS:
            continue
        p, z = xy[sel], h[sel]
        if (np.ptp(p, axis=0) > MAX_EXTENT).any() or np.ptp(z) < MIN_HEIGHT_SPAN:
            continue
        out.append((p.mean(0), int(sel.sum())))
    return out


def track(dets, dt):
    """dets: per sweep list of centroid xy.  Returns list of dict(frames, xy (K, 2))."""
    live, done = [], []
    for i, ds in enumerate(dets):
        pts = np.array([d for d, _ in ds]).reshape(-1, 2)
        used = np.zeros(len(pts), bool)
        pairs = []
        for ti, t in enumerate(live):
            pred = t["xy"][-1] + t["v"] * dt * (i - t["frames"][-1])
            if len(pts):
                d = np.hypot(*(pts - pred).T)
                pairs += [(d[j], ti, j) for j in np.flatnonzero(d < GATE)]
        taken = set()
        for d, ti, j in sorted(pairs):
            if ti in taken or used[j]:
                continue
            t = live[ti]
            gap = i - t["frames"][-1]
            v = (pts[j] - t["xy"][-1]) / (dt * gap)
            t["v"] = 0.5 * t["v"] + 0.5 * v if len(t["frames"]) > 1 else v
            t["frames"].append(i)
            t["xy"].append(pts[j])
            taken.add(ti)
            used[j] = True
        keep = []
        for ti, t in enumerate(live):
            (keep if i - t["frames"][-1] <= MAX_MISSES else done).append(t)
        live = keep + [dict(frames=[i], xy=[pts[j]], v=np.zeros(2)) for j in np.flatnonzero(~used)]
    done += live
    out = []
    for t in done:
        xy = np.array(t["xy"])
        if len(t["frames"]) >= MIN_HITS and np.hypot(*(xy[-1] - xy[0])) >= MIN_TRAVEL:
            out.append(dict(frames=np.array(t["frames"]), xy=xy))
    return out


def segment_tracks(tracks, F, ground_z):
    """Tracks in the layout of hnod.coda.load_segments for a segment of F frames.

    ground_z: (F,) world height of the ground under the walker, to place the boxes.
    """
    out = {}
    for n, t in enumerate(tracks):
        xyz = np.full((F, 3), np.nan)
        yaw = np.full(F, np.nan)
        lwh = np.full((F, 3), np.nan)
        valid = np.zeros(F, bool)
        f, xy = t["frames"], t["xy"]
        heading = np.arctan2(*np.gradient(xy, axis=0)[:, ::-1].T) if len(f) > 1 else np.zeros(len(f))
        xyz[f, :2] = xy
        xyz[f, 2] = ground_z[f] + PERSON_LWH[2] / 2
        yaw[f] = heading
        lwh[f] = PERSON_LWH
        valid[f] = True
        out[f"lidar{n:04d}"] = dict(category="Pedestrian", xyz=xyz, yaw=yaw, lwh=lwh,
                                    occlusion=np.full(F, "Unknown", dtype=object), valid=valid)
    return out
