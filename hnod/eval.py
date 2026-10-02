"""Open-loop collision evaluation of predicted ego trajectories.

The evaluated agent (e.g. a humanoid) is modelled as a vertical cylinder of
radius `radius` whose centre follows the predicted (x, y) path in the scenario
frame.  A prediction is the ego position at each of the N_FUTURE future steps.

Three obstacle sources are checked:
  * dynamic  - tracks that move during the scenario, at their recorded future poses
  * static   - stationary tracks of movable types (standing people, parked bikes and
               cars ...), assumed to stay where they were last seen for the whole horizon
  * map      - occupied cells of the lidar-derived static map
Boxes of STATIC-type tracks (poles, trees, signs, furniture ...) are only used when
the map is not: the map already contains those objects with their true ground-level
footprint, whereas their boxes also enclose overhead parts such as sign plates and
lamp arms.  Obstacles that already overlap the ego at the current step are ignored,
as is the robot operator who walks next to the recording platform.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

from .maps import MAP_RANGE, MAP_RES, PIX_OCCUPIED, PIX_UNKNOWN, xy_to_rowcol
from .scenario import CURRENT, N_FUTURE, PEDESTRIAN, STATIC, wrap

DEFAULT_RADIUS = 0.3     # m, roughly half a humanoid's shoulder width
PEDESTRIAN_RADIUS = 0.3  # m, pedestrians are checked as discs: their labelled boxes are loose (~1 m wide)
OVERHEAD_CLEARANCE = 2.0  # m, boxes whose underside is above this cannot be hit
SUBSTEP_S = 0.1          # paths are checked at least this densely in time


def decode_map(cell):
    """Map column value (PIL image, {'bytes': ...} dict or array) -> uint8 array."""
    if isinstance(cell, dict):
        cell = Image.open(io.BytesIO(cell["bytes"]))
    return np.asarray(cell)


def box_distance(px, py, cx, cy, heading, length, width):
    """Distance from points to oriented rectangles (0 inside). Inputs broadcast."""
    dx, dy = px - cx, py - cy
    c, s = np.cos(heading), np.sin(heading)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - length / 2, 0), np.maximum(np.abs(ly) - width / 2, 0))


def _dense(values, n_sub, angle=False):
    """Linear interpolation of (..., K) step values to n_sub sub-steps per interval.

    Returns (..., (K-1)*n_sub) values at the sub-step ends (the first step itself is excluded).
    """
    a, b = values[..., :-1, None], values[..., 1:, None]
    w = np.arange(1, n_sub + 1) / n_sub
    d = wrap(b - a) if angle else b - a
    out = a + d * w
    return out.reshape(*values.shape[:-1], -1)


def evaluate_scenario(row, pred_xy, radius=DEFAULT_RADIUS, use_map=True, pedestrian_radius=PEDESTRIAN_RADIUS):
    """Score one predicted future path.

    row: a dataset row (dict).  pred_xy: (N_FUTURE, 2) scenario-frame positions
    for steps current+1 .. current+N_FUTURE.  pedestrian_radius: pedestrians are
    treated as discs of this radius around their box centre; None uses their boxes.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    assert pred_xy.shape == (N_FUTURE, 2), pred_xy.shape
    ego, tr = row["ego"], row["tracks"]
    use_map = use_map and row.get("static_map") is not None
    n_sub = max(1, int(round(1.0 / row["rate_hz"] / SUBSTEP_S)))
    path = np.vstack([[ego["x"][CURRENT], ego["y"][CURRENT]], pred_xy])
    px, py = _dense(path[:, 0], n_sub), _dense(path[:, 1], n_sub)  # (S,)
    S = px.size
    res = dict(scenario_id=row["scenario_id"], collided_dynamic=False, collided_static=False, collided_map=False,
               clearance_dynamic=np.inf, clearance_static=np.inf, clearance_map=np.inf,
               first_collision_s=np.nan, start_overlap=False, unknown_fraction=0.0)
    hit_time = np.full(S, False)

    if len(tr["id"]):
        x, y, hd = (np.asarray(tr[k], dtype=np.float64) for k in ("x", "y", "heading"))
        valid = np.asarray(tr["valid"], dtype=bool)
        L, W = np.asarray(tr["length"], dtype=np.float64)[:, None], np.asarray(tr["width"], dtype=np.float64)[:, None]
        if pedestrian_radius is not None:
            # A zero-size box makes box_distance the centre distance; the disc radius is added to the agent's below.
            ped = np.asarray(tr["object_type"]) == PEDESTRIAN
            L[ped], W[ped] = 0.0, 0.0
            reach = radius + np.where(ped, pedestrian_radius, 0.0)
        else:
            reach = np.full(len(L), radius)
        z, H = np.asarray(tr["z"], dtype=np.float64), np.asarray(tr["height"])
        with np.errstate(all="ignore"):
            underside = np.nanmin(np.where(valid, z, np.nan), axis=1) - H / 2
        reachable = ~np.asarray(tr["is_operator"]) & (underside < OVERHEAD_CLEARANCE)
        if use_map:
            reachable &= np.asarray(tr["object_type"]) != STATIC
        stationary = np.asarray(tr["is_stationary"], dtype=bool)

        # --- stationary tracks: frozen at the valid step closest to the current one
        idx = np.flatnonzero(stationary & reachable)
        if len(idx):
            order = np.argsort(np.abs(np.arange(valid.shape[1]) - CURRENT))
            ref = order[np.argmax(valid[idx][:, order], axis=1)]
            sx, sy, sh = x[idx, ref], y[idx, ref], hd[idx, ref]
            start = box_distance(path[0, 0], path[0, 1], sx, sy, sh, L[idx, 0], W[idx, 0]) <= reach[idx]
            res["start_overlap"] |= bool(start.any())
            gap = (box_distance(px[:, None], py[:, None], sx, sy, sh, L[idx, 0], W[idx, 0]) - reach[idx])[:, ~start]
            if gap.size:
                res["clearance_static"] = float(gap.min())
                res["collided_static"] = bool((gap <= 0).any())
                hit_time |= (gap <= 0).any(1)

        # --- moving tracks: interpolated between consecutive valid future steps
        idx = np.flatnonzero(~stationary & reachable)
        if len(idx):
            fut = slice(CURRENT, None)
            v = valid[idx][:, fut]
            ok = np.repeat(v[:, :-1] & v[:, 1:], n_sub, axis=1)
            ok[:, n_sub - 1::n_sub] = v[:, 1:]  # a sub-step that lands on a step only needs that step
            ax = _dense(np.nan_to_num(x[idx][:, fut]), n_sub)
            ay = _dense(np.nan_to_num(y[idx][:, fut]), n_sub)
            ah = _dense(np.nan_to_num(hd[idx][:, fut]), n_sub, angle=True)
            # a sub-step landing exactly on a valid step must use that step's own pose
            ax[:, n_sub - 1::n_sub], ay[:, n_sub - 1::n_sub] = x[idx][:, fut][:, 1:], y[idx][:, fut][:, 1:]
            ah[:, n_sub - 1::n_sub] = hd[idx][:, fut][:, 1:]
            with np.errstate(invalid="ignore"):
                start = valid[idx, CURRENT] & (box_distance(path[0, 0], path[0, 1], x[idx, CURRENT], y[idx, CURRENT],
                                                            hd[idx, CURRENT], L[idx, 0], W[idx, 0]) <= reach[idx])
            res["start_overlap"] |= bool(start.any())
            ok &= ~start[:, None]
            gap = box_distance(px[None], py[None], np.nan_to_num(ax), np.nan_to_num(ay), np.nan_to_num(ah),
                               L[idx], W[idx]) - reach[idx, None]
            gap = np.where(ok, gap, np.inf)
            if np.isfinite(gap).any():
                res["clearance_dynamic"] = float(gap.min())
                res["collided_dynamic"] = bool((gap <= 0).any())
                hit_time |= (gap <= 0).any(0)

    if use_map:
        img = decode_map(row["static_map"])
        occ = img == PIX_OCCUPIED
        rr, cc = np.indices(img.shape)
        r0, c0 = xy_to_rowcol(path[0, 0], path[0, 1])
        near_start = np.hypot(rr - r0, cc - c0) * MAP_RES <= radius + MAP_RES
        res["start_overlap"] |= bool((occ & near_start).any())
        occ &= ~near_start
        dist = ndimage.distance_transform_edt(~occ) * MAP_RES if occ.any() else np.full(img.shape, np.inf)
        r, c = xy_to_rowcol(px, py)
        r, c = np.round(r).astype(int), np.round(c).astype(int)
        inside = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])
        if inside.any():
            d = dist[r[inside], c[inside]]
            res["clearance_map"] = float(d.min() - radius)
            res["collided_map"] = bool((d <= radius).any())
            hit_time[np.flatnonzero(inside)[d <= radius]] = True
            res["unknown_fraction"] = float((img[r[inside], c[inside]] == PIX_UNKNOWN).mean())

    res["collided"] = res["collided_dynamic"] or res["collided_static"] or res["collided_map"]
    if hit_time.any():
        res["first_collision_s"] = float((np.argmax(hit_time) + 1) / (n_sub * row["rate_hz"]))
    expert = np.stack([ego["x"], ego["y"]], 1)[CURRENT + 1:]
    err = np.linalg.norm(pred_xy - expert, axis=1)
    res["ade"], res["fde"] = float(err.mean()), float(err[-1])
    return res


def aggregate(results):
    """Mean metrics over a list of evaluate_scenario outputs."""
    keys = ["collided", "collided_dynamic", "collided_static", "collided_map", "start_overlap",
            "unknown_fraction", "ade", "fde"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["num_scenarios"] = len(results)
    return out


# ----------------------------------------------------------------------------- baselines

def baseline_expert(row):
    """The recorded robot's own future (teleoperated by a human)."""
    return np.stack([row["ego"]["x"], row["ego"]["y"]], 1)[CURRENT + 1:]


def baseline_constant_velocity(row):
    v = np.array([row["ego"]["vx"][CURRENT], row["ego"]["vy"][CURRENT]])
    t = np.arange(1, N_FUTURE + 1) / row["rate_hz"]
    return t[:, None] * v[None]


def baseline_stationary(row):
    return np.zeros((N_FUTURE, 2))


def baseline_straight_to_goal(row):
    """Constant-speed straight line from the current position to the goal."""
    return np.arange(1, N_FUTURE + 1)[:, None] / N_FUTURE * np.asarray(row["goal"])[None]


BASELINES = {"expert": baseline_expert, "constant_velocity": baseline_constant_velocity,
             "stationary": baseline_stationary, "straight_to_goal": baseline_straight_to_goal}
