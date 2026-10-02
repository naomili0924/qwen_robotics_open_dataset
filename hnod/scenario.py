"""Dataset-agnostic segment clean-up and scenario windowing.

A *segment* is a stretch of consecutively labelled frames with world-frame ego
poses and tracked boxes (see hnod.coda.load_segments for the layout).  A
*scenario* is a fixed window of N_PAST + 1 + N_FUTURE steps cut from a segment
and expressed in an ego-centric frame, mirroring a Waymo Open Motion scenario.
"""
import numpy as np

N_PAST = 10
N_FUTURE = 10
N_STEPS = N_PAST + 1 + N_FUTURE
CURRENT = N_PAST

PEDESTRIAN, CYCLE, VEHICLE, OTHER_MOVABLE, STATIC = "PEDESTRIAN", "CYCLE", "VEHICLE", "OTHER_MOVABLE", "STATIC"
OCCLUSION_CODES = {"None": 0, "Light": 1, "Medium": 2, "Heavy": 3, "Full": 4, "Unknown": 5}

# A track is cut where consecutive observations are further apart than
# base + speed * dt: source datasets occasionally reuse an instance id for a
# different object.
_JUMP_LIMITS = {PEDESTRIAN: (1.0, 4.0), CYCLE: (1.5, 15.0), VEHICLE: (1.5, 15.0),
                OTHER_MOVABLE: (1.5, 15.0), STATIC: (1.0, 0.0)}
MAX_BOX_RANGE = 150.0   # boxes further than this from the robot are label garbage
MAX_BOX_DIM = 40.0
STATIONARY_RADIUS = 0.25  # a track that stays within this of its median position is stationary (segments may override)
STATIONARY_MIN_STEPS = 3   # ... and only if it is seen at least this many steps
STATIONARY_MIN_S = 1.0     # ... spanning at least this long
VELOCITY_HALF_WINDOW_S = 0.3  # velocities are central differences over up to +- this long (at least one frame)
# The robot's human operator (sometimes two people) walks right next to it for
# whole recordings.  Within OPERATOR_CONTEXT_S of a scenario's current step, a
# pedestrian track counts as operator if it stays within OPERATOR_RADIUS of the
# robot and either lasts OPERATOR_LONG_S, or lasts OPERATOR_SHORT_S while the
# robot travels OPERATOR_MIN_TRAVEL (i.e. it keeps up).  The test is local in
# time because instance ids get reused for other people later in a recording.
OPERATOR_CONTEXT_S = 10.0
OPERATOR_RADIUS = 2.5
OPERATOR_LONG_S = 5.0
OPERATOR_SHORT_S = 3.0
OPERATOR_MIN_TRAVEL = 3.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def finite_diff_velocity(xy, valid, ts, k_max=3):
    """Velocity by central differences over the widest available baseline (<= k_max frames)."""
    F = len(ts)
    v = np.zeros((F, 2))
    done = np.zeros(F, bool)
    for k in range(k_max, 0, -1):
        a = np.arange(k, F - k)
        i = a[valid[a - k] & valid[a + k] & valid[a] & ~done[a]]
        v[i] = (xy[i + k] - xy[i - k]) / (ts[i + k] - ts[i - k])[:, None]
        done[i] = True
    for k in range(1, k_max + 1):
        for sign in (1, -1):
            a = np.arange(0, F - k) if sign > 0 else np.arange(k, F)
            i = a[valid[a] & valid[a + sign * k] & ~done[a]]
            v[i] = (xy[i + sign * k] - xy[i]) / (ts[i + sign * k] - ts[i])[:, None]
            done[i] = True
    return v


def _interpolate_unlabelled(seg):
    """Fill single unlabelled frames inside a segment from their two neighbours."""
    missing = np.flatnonzero(~seg["labelled"])
    for i in missing:
        if i == 0 or i == len(seg["labelled"]) - 1:
            continue
        for tr in seg["tracks"].values():
            if tr["valid"][i - 1] and tr["valid"][i + 1]:
                tr["xyz"][i] = (tr["xyz"][i - 1] + tr["xyz"][i + 1]) / 2
                tr["yaw"][i] = tr["yaw"][i - 1] + wrap(tr["yaw"][i + 1] - tr["yaw"][i - 1]) / 2
                tr["lwh"][i] = tr["lwh"][i - 1]
                tr["occlusion"][i] = tr["occlusion"][i - 1]
                tr["valid"][i] = True


def clean_segment(seg, type_of):
    """Sanitise tracks and stack them into dense arrays.

    type_of: callable mapping a source category name to one of the coarse types.
    Adds to seg: track_ids, categories, types (N,), dims (N,3), xyz (N,F,3),
    yaw/valid/occlusion (N,F), vel (N,F,2), near_ego (N,F), ego_xyz, ego_yaw, ego_vel.
    """
    _interpolate_unlabelled(seg)
    ts = seg["timestamps"]
    F = len(ts)
    k_max = max(1, int(round(VELOCITY_HALF_WINDOW_S * seg["rate_hz"])))
    ego_xyz = seg["ego_T"][:, :3, 3]
    ego_yaw = yaw_of(seg["ego_T"][:, :3, :3])
    out = []
    for iid, tr in seg["tracks"].items():
        typ = type_of(tr["category"])
        valid = tr["valid"].copy()
        rng = np.linalg.norm(tr["xyz"][:, :2] - ego_xyz[:, :2], axis=1)
        with np.errstate(invalid="ignore"):
            sane = np.isfinite(tr["xyz"]).all(1) & (rng < MAX_BOX_RANGE) & (tr["lwh"] > 0).all(1) \
                & (tr["lwh"] < MAX_BOX_DIM).all(1)
        valid &= sane
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            continue
        base, speed = _JUMP_LIMITS[typ]
        step = np.linalg.norm(np.diff(tr["xyz"][idx, :2], axis=0), axis=1)
        cuts = np.flatnonzero(step > base + speed * np.diff(ts[idx])) + 1
        for part, sub in enumerate(np.split(idx, cuts)):
            v = np.zeros(F, bool)
            v[sub] = True
            out.append(dict(id=iid if part == 0 else f"{iid}#{part + 1}", category=tr["category"], type=typ,
                            valid=v, xyz=tr["xyz"], yaw=tr["yaw"], lwh=tr["lwh"], occlusion=tr["occlusion"]))

    N = len(out)
    seg["track_ids"] = np.array([t["id"] for t in out], dtype=object)
    seg["categories"] = np.array([t["category"] for t in out], dtype=object)
    seg["types"] = np.array([t["type"] for t in out], dtype=object)
    seg["valid"] = np.stack([t["valid"] for t in out]) if N else np.zeros((0, F), bool)
    seg["xyz"] = np.full((N, F, 3), np.nan)
    seg["yaw"] = np.full((N, F), np.nan)
    seg["occlusion"] = np.full((N, F), -1, np.int8)
    seg["dims"] = np.zeros((N, 3))
    seg["vel"] = np.zeros((N, F, 2))
    seg["near_ego"] = np.zeros((N, F), bool)
    for n, t in enumerate(out):
        v = t["valid"]
        seg["xyz"][n, v] = t["xyz"][v]
        seg["yaw"][n, v] = wrap(t["yaw"][v])
        seg["occlusion"][n, v] = [OCCLUSION_CODES.get(o, 5) for o in t["occlusion"][v]]
        seg["dims"][n] = np.median(t["lwh"][v], axis=0)
        seg["vel"][n] = finite_diff_velocity(np.nan_to_num(seg["xyz"][n, :, :2]), v, ts, k_max)
        seg["near_ego"][n, v] = np.linalg.norm(seg["xyz"][n, v, :2] - ego_xyz[v, :2], axis=1) < OPERATOR_RADIUS
    seg["ego_xyz"], seg["ego_yaw"] = ego_xyz, ego_yaw
    seg["ego_vel"] = finite_diff_velocity(ego_xyz[:, :2], np.ones(F, bool), ts, k_max)
    return seg


def window_anchors(seg, step, stride):
    """Segment-frame indices usable as the current step of a scenario.

    Segments start and end on labelled frames and interior label gaps are
    interpolated by clean_segment, so every in-range window is usable.
    """
    F = len(seg["timestamps"])
    return list(range(N_PAST * step, F - N_FUTURE * step, stride))


def find_operators(seg, anchor, rows):
    """Which of the track rows are the robot's operator around segment frame `anchor`."""
    rate = seg["rate_hz"]
    c = int(OPERATOR_CONTEXT_S * rate)
    w = slice(max(0, anchor - c), anchor + c + 1)
    valid, near = seg["valid"][rows, w], seg["near_ego"][rows, w]
    n_valid = valid.sum(1)
    hop = np.r_[0.0, np.linalg.norm(np.diff(seg["ego_xyz"][w, :2], axis=0), axis=1)]
    travel = (valid * hop).sum(1)  # distance the robot covers while the track is visible
    keeps_up = (n_valid >= OPERATOR_LONG_S * rate) | ((n_valid >= OPERATOR_SHORT_S * rate) & (travel >= OPERATOR_MIN_TRAVEL))
    return (seg["types"][rows] == PEDESTRIAN) & keeps_up & (near.sum(1) > 0.9 * np.maximum(n_valid, 1))


def build_scenario(seg, anchor, step):
    """Cut one ego-centric scenario around segment frame `anchor`.

    The scenario frame is the robot's own frame at the current step, moved down to
    the ground: origin on the ground under the sensor/ego origin, +x forward, +y
    left, +z along the robot's up axis.  It is a rigid transform of the source world
    frame (`world_from_scenario`), so lidar points and camera poses map into it exactly.
    Headings and velocities are rotated by the ego yaw only, which ignores the
    robot's tilt (a few degrees at most).
    """
    idx = anchor + step * np.arange(-N_PAST, N_FUTURE + 1)
    yaw0 = seg["ego_yaw"][anchor]
    R_cur, t_cur = seg["ego_T"][anchor, :3, :3], seg["ego_T"][anchor, :3, 3]
    lift = np.array([0.0, 0.0, seg["lidar_height"]])
    c, s = np.cos(yaw0), np.sin(yaw0)
    R = np.array([[c, s], [-s, c]])  # world -> scenario (2D), for headings and velocities

    def to_local(xyz):
        return (xyz - t_cur) @ R_cur + lift

    L, W, H = seg["ego_size"]
    ego_local = to_local(seg["ego_xyz"][idx].copy())
    ego_local[:, 2] += H / 2 - seg["lidar_height"]
    ego_vel = seg["ego_vel"][idx] @ R.T
    ego = dict(x=ego_local[:, 0], y=ego_local[:, 1], z=ego_local[:, 2], heading=wrap(seg["ego_yaw"][idx] - yaw0),
               vx=ego_vel[:, 0], vy=ego_vel[:, 1], length=L, width=W, height=H)

    valid = seg["valid"][:, idx]
    keep = np.flatnonzero(valid[:, CURRENT:].any(1))  # only objects present now or later are published
    valid = valid[keep]
    xyz = to_local(seg["xyz"][keep][:, idx].copy())
    vel = seg["vel"][keep][:, idx] @ R.T
    vel[~valid] = np.nan
    heading = wrap(seg["yaw"][keep][:, idx] - yaw0)
    types = seg["types"][keep]

    # Stationary: static classes always; others if they are seen staying put inside the
    # window.  One or two sightings are not evidence of that, so they do not count.
    xy = xyz[..., :2]
    with np.errstate(all="ignore"):
        med = np.nanmedian(xy, axis=1, keepdims=True)
        spread = np.nanmax(np.linalg.norm(xy - med, axis=2), axis=1)
    first, last = valid.argmax(1), valid.shape[1] - 1 - valid[:, ::-1].argmax(1)
    seen_long = (valid.sum(1) >= STATIONARY_MIN_STEPS) & ((last - first) * step / seg["rate_hz"] >= STATIONARY_MIN_S)
    is_stationary = (types == STATIC) | (seen_long & (spread < seg.get("stationary_radius", STATIONARY_RADIUS)))
    is_operator = find_operators(seg, anchor, keep)
    movable = types != STATIC
    to_predict = np.flatnonzero(movable & ~is_operator & valid[:, CURRENT] & valid[:, CURRENT + 1:].any(1))

    T = seg["ego_T"][anchor].copy()
    T[:3, 3] = t_cur - R_cur @ lift
    tracks = dict(id=seg["track_ids"][keep], category=seg["categories"][keep], object_type=types,
                  length=seg["dims"][keep, 0], width=seg["dims"][keep, 1], height=seg["dims"][keep, 2],
                  is_stationary=is_stationary, is_operator=is_operator,
                  x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2], heading=heading, vx=vel[..., 0], vy=vel[..., 1],
                  valid=valid, occlusion=seg["occlusion"][keep][:, idx])
    rate = seg["rate_hz"] / step
    frame0 = int(seg["frames"][anchor])
    return dict(
        scenario_id=f"{seg['dataset']}_{seg['sequence']:0>2}_{frame0:06d}_{rate:g}hz",
        dataset=seg["dataset"], sequence=seg["sequence"], segment=int(seg["segment"]),
        rate_hz=float(rate), current_time_index=CURRENT,
        timestamps=seg["timestamps"][idx], source_frames=seg["frames"][idx],
        world_from_scenario=T, ego=ego, goal=ego_local[-1, :2].copy(), tracks=tracks,
        tracks_to_predict=to_predict, segment_index=idx, track_rows=keep)
