"""Evaluation suite: scoring a predicted path, deriving goals and tagging scenarios.

Protocol (docs/eval_design.md): the model sees the past images, its past poses and a text prompt
that contains the goal; it returns a path (x, y), metres in the robot frame at the current moment,
of any length.  The robot is assumed to follow that path at ``speed`` m/s with perfect control
(a slow humanoid walker, 0.5 m/s by default); a path that ends early means the robot stops there.
Collisions are then checked with ``hnod.eval`` against recorded people, objects, the static map
and lidar; progress, efficiency and smoothness are computed from the path itself, and distance to
the reference path (a human-driven robot or a shortest-path planner) is reported separately.
"""
import numpy as np

from . import eval as ev
from .scenario import CURRENT, N_FUTURE, PEDESTRIAN

ROBOT_SPEED = 0.5      # m/s
GOAL_RADIUS = 0.5      # m, success radius where the horizon is long enough to reach the goal
MAP_TOLERANCE = 0.05   # m, half a static-map cell: grazing a wall at exactly the robot radius is not a collision


def _arc(path):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]


def resample_by_distance(path, spacing, length=None):
    """Points every `spacing` metres along a path that starts at its first point."""
    s = _arc(path)
    total = s[-1] if length is None else min(length, s[-1])
    q = np.arange(spacing, total + 1e-9, spacing)
    if not len(q):
        return path[:1].copy()
    return np.stack([np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])], 1)


def timed_positions(pred_xy, rate_hz, speed=ROBOT_SPEED, n=N_FUTURE):
    """Positions at the scenario's future steps when the path is followed at `speed` from (0, 0)."""
    pred = np.asarray(pred_xy, float).reshape(-1, 2)
    path = np.vstack([[0.0, 0.0], pred])
    s = _arc(path)
    d = np.minimum(speed * np.arange(1, n + 1) / rate_hz, s[-1])
    if s[-1] <= 0:
        return np.zeros((n, 2))
    return np.stack([np.interp(d, s, path[:, 0]), np.interp(d, s, path[:, 1])], 1)


def wiggle(path, step=0.1):
    """Heading oscillation in radians: total turning minus net turning, on the path resampled every `step` m.

    A turn in place or one smooth curve scores 0; zig-zags score high.
    """
    p = resample_by_distance(np.vstack([[0.0, 0.0], path]), step)
    p = np.vstack([[0.0, 0.0], p])
    if len(p) < 3:
        return 0.0
    h = np.unwrap(np.arctan2(*np.diff(p, axis=0)[:, ::-1].T))
    dh = np.diff(h)
    return float(np.abs(dh).sum() - abs(dh.sum()))


def at_fault_dynamic(row, steps, radius=ev.DEFAULT_RADIUS, pedestrian_radius=ev.PEDESTRIAN_RADIUS):
    """Collisions with moving agents that the robot causes.

    Recorded people do not react to the robot.  A slow robot is therefore walked into from behind by
    people who followed the recording platform, and a standing robot is walked through.  As in nuPlan's
    at-fault collisions, a contact counts only if the robot is moving at that moment and the agent's
    centre is in front of it (ahead of its direction of motion).  Returns (collided, clearance).
    """
    tr = ev.future_tracks(row)
    if not len(tr["id"]):
        return False, np.inf
    n_sub = max(1, int(round(1.0 / row["rate_hz"] / ev.SUBSTEP_S)))
    path = np.vstack([[0.0, 0.0], steps])
    px, py = ev._dense(path[:, 0], n_sub), ev._dense(path[:, 1], n_sub)
    mx, my = np.gradient(px), np.gradient(py)
    moving = np.hypot(mx, my) * n_sub * row["rate_hz"] > 0.05
    x, y, hd = (np.asarray(tr[k], dtype=np.float64) for k in ("x", "y", "heading"))
    valid = np.asarray(tr["valid"], dtype=bool)
    L, W = np.asarray(tr["length"], float)[:, None], np.asarray(tr["width"], float)[:, None]
    ped = np.asarray(tr["object_type"]) == PEDESTRIAN
    L[ped], W[ped] = 0.0, 0.0
    reach = radius + np.where(ped, pedestrian_radius, 0.0)
    z, H = np.asarray(tr["z"], float), np.asarray(tr["height"])
    with np.errstate(all="ignore"):
        underside = np.nanmin(np.where(valid, z, np.nan), axis=1) - H / 2
    idx = np.flatnonzero(~np.asarray(tr["is_stationary"], bool) & ~np.asarray(tr["is_operator"], bool)
                         & (underside < ev.OVERHEAD_CLEARANCE) & (np.asarray(tr["object_type"]) != ev.STATIC))
    if not len(idx):
        return False, np.inf
    v = valid[idx]
    ok = np.repeat(v[:, :-1] & v[:, 1:], n_sub, axis=1)
    ok[:, n_sub - 1::n_sub] = v[:, 1:]
    ax, ay = ev._dense(np.nan_to_num(x[idx]), n_sub), ev._dense(np.nan_to_num(y[idx]), n_sub)
    ah = ev._dense(np.nan_to_num(hd[idx]), n_sub, angle=True)
    ax[:, n_sub - 1::n_sub], ay[:, n_sub - 1::n_sub] = x[idx][:, 1:], y[idx][:, 1:]
    ah[:, n_sub - 1::n_sub] = hd[idx][:, 1:]
    with np.errstate(invalid="ignore"):
        start = valid[idx, 0] & (ev.box_distance(0.0, 0.0, x[idx, 0], y[idx, 0], hd[idx, 0], L[idx, 0],
                                                 W[idx, 0]) <= reach[idx])
    ok &= ~start[:, None]
    gap = ev.box_distance(px[None], py[None], ax, ay, ah, L[idx], W[idx]) - reach[idx, None]
    gap = np.where(ok, gap, np.inf)
    ahead = (ax - px[None]) * mx[None] + (ay - py[None]) * my[None] > 0
    contact = gap <= 0
    fault = False
    for a in np.flatnonzero(contact.any(1)):  # judged at the first moment of contact with each agent
        k = int(np.argmax(contact[a]))
        fault |= bool(ahead[a, k] and moving[k])
    return fault, float(gap.min()) if np.isfinite(gap).any() else np.inf


def score(row, pred_xy, speed=ROBOT_SPEED, radius=ev.DEFAULT_RADIUS):
    """All metrics for one scenario and one predicted path."""
    rate, goal = row["rate_hz"], np.asarray(row["goal"], float)
    steps = timed_positions(pred_xy, rate, speed)
    res = ev.evaluate_scenario(row, steps, radius=radius, map_tolerance=MAP_TOLERANCE)
    res["collided_dynamic_any"] = res["collided_dynamic"]  # including being walked into (not counted)
    res["collided_dynamic"], res["clearance_dynamic"] = at_fault_dynamic(row, steps, radius)
    res["collided"] = res["collided_dynamic"] or res["collided_static"] or res["collided_map"]
    horizon_s = N_FUTURE / rate
    d0, d1 = np.hypot(*goal), np.hypot(*(goal - steps[-1]))
    reachable = min(speed * horizon_s, d0)  # the most progress a perfect path could make
    travelled = float(_arc(np.vstack([[0.0, 0.0], steps]))[-1])
    res.update(
        progress_m=float(d0 - d1),
        progress_ratio=float((d0 - d1) / reachable) if reachable > 0 else 0.0,  # 1 = straight at full speed
        path_efficiency=float((d0 - d1) / travelled) if travelled > 1e-3 else 0.0,
        reached_goal=bool(d1 <= GOAL_RADIUS),
        wiggle_rad=wiggle(np.asarray(pred_xy, float).reshape(-1, 2)),
    )
    ref = np.stack([row["ego"]["x"], row["ego"]["y"]], 1)[CURRENT:]
    ref = ref - ref[0]
    n = max(1, int(round(speed * horizon_s / 0.25)))
    a = resample_by_distance(np.vstack([[0.0, 0.0], steps]), 0.25)[:n]
    b = resample_by_distance(ref, 0.25)[:n]
    m = min(len(a), len(b))
    res["reference_deviation_m"] = float(np.linalg.norm(a[:m] - b[:m], axis=1).mean()) if m else float("nan")
    res["success"] = bool(not res["collided"] and res["progress_ratio"] >= 0.5)
    return res


METRICS = ["success", "collided", "collided_dynamic", "collided_static", "collided_map", "collided_lidar",
           "progress_ratio", "path_efficiency", "wiggle_rad", "reference_deviation_m", "unknown_fraction"]


def aggregate(results, keys=METRICS):
    out = {k: float(np.nanmean([float(r[k]) for r in results])) for k in keys}
    out["num_scenarios"] = len(results)
    return out


# ----------------------------------------------------------------------------- baselines in the suite protocol

def baseline_paths(row):
    """Paths (not timed) for the naive baselines; each is followed at the robot speed."""
    goal = np.asarray(row["goal"], float)
    v = np.array([row["ego"]["vx"][CURRENT], row["ego"]["vy"][CURRENT]])
    heading = v / np.linalg.norm(v) if np.linalg.norm(v) > 0.1 else np.array([1.0, 0.0])
    far = 10.0
    return {"stationary": np.zeros((1, 2)),
            "straight_ahead": heading[None] * np.linspace(0.25, far, 40)[:, None],
            "straight_to_goal": goal[None] * np.linspace(0, 1, 41)[1:, None]}


# ----------------------------------------------------------------------------- tags

def people_tags(row, ref_xy, near=8.0):
    """Tags from the recorded people around the reference path."""
    tr = row["future_tracks"]
    if not len(tr["id"]):
        return ["people:none"], 0
    ped = (np.asarray(tr["object_type"]) == PEDESTRIAN) & ~np.asarray(tr["is_operator"], bool)
    valid = np.asarray(tr["valid"], bool)
    x, y = np.asarray(tr["x"], float), np.asarray(tr["y"], float)
    now = ped & valid[:, 0] & (np.hypot(x[:, 0], y[:, 0]) < near)
    n = int(now.sum())
    tags = ["people:none" if n == 0 else "people:crowd" if n >= 5 else "people:few"]
    stationary = np.asarray(tr["is_stationary"], bool)
    ref = np.asarray(ref_xy, float)  # (N_FUTURE + 1, 2) from the current step
    ref_dir = ref[-1] - ref[0]
    for i in np.flatnonzero(ped & ~stationary):
        ok = valid[i]
        if ok.sum() < 2:
            continue
        d = np.hypot(x[i, ok] - ref[ok, 0], y[i, ok] - ref[ok, 1])  # same-time distance to the reference
        if d.min() > 1.5:
            continue
        idx = np.flatnonzero(ok)
        mv = np.array([x[i, idx[-1]] - x[i, idx[0]], y[i, idx[-1]] - y[i, idx[0]]])
        if np.linalg.norm(mv) < 0.5 or np.linalg.norm(ref_dir) < 0.5:
            continue
        cos = mv @ ref_dir / np.linalg.norm(mv) / np.linalg.norm(ref_dir)
        tags.append("people:oncoming" if cos < -0.7 else "people:same_direction" if cos > 0.7 else "people:crossing")
    if (ped & stationary & valid[:, 0] & (np.hypot(x[:, 0], y[:, 0]) < 4.0)).any():
        tags.append("people:standing_nearby")
    return sorted(set(tags)), n


def path_tags(ref_xy, goal, clearance=None):
    ref = np.asarray(ref_xy, float)
    tags = []
    seg = np.diff(ref, axis=0)
    moving = np.hypot(*seg.T) > 0.05
    if moving.sum() >= 2:
        h = np.unwrap(np.arctan2(seg[moving, 1], seg[moving, 0]))
        turn = abs(h[-1] - h[0])
        if turn > np.radians(135):
            tags.append("path:u_turn")
        elif turn > np.radians(45):
            tags.append("path:turn")
        else:
            tags.append("path:straight")
    else:
        tags.append("path:mostly_still")
    bearing = np.degrees(abs(np.arctan2(goal[1], goal[0])))
    tags.append("goal:ahead" if bearing <= 30 else "goal:side" if bearing <= 90 else "goal:behind")
    if clearance is not None and np.isfinite(clearance) and clearance < 0.35:
        tags.append("layout:narrow")
    return tags


# ----------------------------------------------------------------------------- final-frame task (positions in time)

COMPLETE_RADIUS = 1.0  # m, the predicted end must be this close to where the recording ended


def steps_from_timed(pred_xy, final_step, n=N_FUTURE):
    """Model output (K positions evenly spaced in time up to the final frame) -> positions at the scenario's
    future steps.  Steps after the final frame repeat the last position (the robot has arrived and stands)."""
    pred = np.asarray(pred_xy, float).reshape(-1, 2)
    path = np.vstack([[0.0, 0.0], pred])
    u = np.linspace(0.0, 1.0, len(path))                      # fraction of the horizon at each given position
    q = np.minimum(np.arange(1, n + 1) / final_step, 1.0)     # fraction at each scenario step
    return np.stack([np.interp(q, u, path[:, 0]), np.interp(q, u, path[:, 1])], 1)


def score_timed(row, pred_xy, radius=ev.DEFAULT_RADIUS):
    """Metrics for the final-frame task: the prediction says where the robot is at each moment, so collisions
    are checked at the recorded timing (no assumed speed).  row needs final_step and final_xy."""
    step = int(row["final_step"])
    steps = steps_from_timed(pred_xy, step)
    res = ev.evaluate_scenario(row, steps, radius=radius, map_tolerance=MAP_TOLERANCE)
    res["collided_dynamic_any"] = res["collided_dynamic"]
    res["collided_dynamic"], res["clearance_dynamic"] = at_fault_dynamic(row, steps, radius)
    res["collided"] = res["collided_dynamic"] or res["collided_static"] or res["collided_map"]
    e = row["ego"]
    ref = np.stack([e["x"], e["y"]], 1)[CURRENT + 1:CURRENT + 1 + step] - [e["x"][CURRENT], e["y"][CURRENT]]
    err = np.linalg.norm(steps[:step] - ref, axis=1)
    dt = 1.0 / row["rate_hz"]
    vel = np.diff(np.vstack([[0.0, 0.0], steps[:step]]), axis=0) / dt
    acc = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt if step > 1 else np.zeros(1)
    end_error = float(np.linalg.norm(steps[step - 1] - np.asarray(row["final_xy"], float)))
    res.update(ade=float(err.mean()), fde=float(err[-1]), end_error_m=end_error,
               completed=bool(end_error <= COMPLETE_RADIUS), wiggle_rad=wiggle(steps[:step]),
               max_accel=float(acc.max()), speed_mps=float(np.linalg.norm(vel, axis=1).mean()),
               reference_distance_m=float(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], ref]), axis=0), axis=1).sum()))
    res["success"] = bool(res["completed"] and not res["collided"])
    return res


TIMED_METRICS = ["success", "completed", "collided", "collided_dynamic", "collided_map", "ade", "fde", "end_error_m",
                 "wiggle_rad", "max_accel", "speed_mps"]


def timed_baselines(row, k=10):
    """K positions over the horizon for the naive planners: stand still, keep the current velocity, the recording."""
    step, e, rate = int(row["final_step"]), row["ego"], row["rate_hz"]
    t = np.arange(1, k + 1) / k * step / rate
    v = np.array([e["vx"][CURRENT], e["vy"][CURRENT]])
    ref = np.stack([e["x"], e["y"]], 1)[CURRENT:CURRENT + 1 + step] - [e["x"][CURRENT], e["y"][CURRENT]]
    u = np.arange(step + 1) / step
    q = np.arange(1, k + 1) / k
    return {"stationary": np.zeros((k, 2)), "constant_velocity": t[:, None] * v[None],
            "recorded": np.stack([np.interp(q, u, ref[:, 0]), np.interp(q, u, ref[:, 1])], 1)}
