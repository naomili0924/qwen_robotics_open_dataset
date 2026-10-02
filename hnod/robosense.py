"""Reader for RoboSense (Su et al., CVPR 2025) annotation files.

Source layout (Hugging Face `suhaisheng0527/RoboSense`):
  splits/robosense_global_{train,val}.pkl   list of labelled frames, 1 Hz
Each frame holds boxes in the ego frame (origin at the rear axle, near the ground;
x forward, y left, z up), the ego pose in a global ENU frame and the path/pose of
the Hesai lidar sweep.  Conventions established from the data:
  * annos.location is the box *bottom* centre, annos.dimensions is (w, l, h)
  * heading = rotation_y + pi/2 (and is front/back ambiguous in the labels)
  * `annos.id` is only a hint: ids restart in every ~20 s clip (`seq_token`), 6.5%
    of frames contain the same id twice, and 2.5% of id links imply impossible speeds

Clips are consecutive 20 s cuts of longer recordings.  A 21-step scenario at
1 Hz needs more than one clip, so clips that follow each other without a gap
(and belong to the same split) are chained.  Tracks are then built by
frame-to-frame association on position, which uses a label id only when the
motion it implies is physically possible.
"""
import collections
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

RATE_HZ = 1.0
MAX_STEP_S = 1.5          # frames further apart than this break a chain
MAX_EGO_SPEED = 3.0       # m/s; the platform drives below 1 m/s, faster means a pose glitch
# The platform is a small road sweeper; its size is not published.  Nominal box.
EGO_SIZE = (1.5, 0.9, 1.4)
STATIONARY_RADIUS = 0.5   # poses/labels jitter more than CODa's; parked cars wander ~0.3 m

# Frame-to-frame association.  An object continues a track when it lies within
# GATE (+ 0.5 m per m/s of track speed) of the track's constant-velocity prediction.
# If both carry the same label id the limit is looser: the prediction may be off by
# what MAX_ACCEL allows, or, for a track seen only once (no velocity yet), the
# object may be anywhere MAX_SPEED could have taken it.
_GATE = {"Pedestrian": 1.5, "Cyclist": 3.0, "Car": 2.5}          # m
_MAX_SPEED = {"Pedestrian": 3.0, "Cyclist": 10.0, "Car": 16.0}   # m/s
_MAX_ACCEL = {"Pedestrian": 1.5, "Cyclist": 3.0, "Car": 4.0}     # m/s^2
_ID_BONUS = 0.1          # cost factor for id-consistent links, so they win ties
MAX_MISSED = 3           # a track unseen for up to this many frames can be picked up again ...
REACQUIRE_RADIUS = 1.0   # ... by an object this close to where it was last seen [m]
_MOVING = {"Pedestrian": 0.7, "Cyclist": 2.0, "Car": 2.0}  # speed above which heading can be checked [m/s]


def load_frames(pkl_dir, split):
    with open(Path(pkl_dir) / f"robosense_global_{split}.pkl", "rb") as f:
        return pickle.load(f)


def recording_of(frame):
    """Recording batch a frame belongs to (the top-level directory of its sweep)."""
    return frame["hs64_path"].split("/")[1]


def ego_origin_heights(frames):
    """Height of the ego-frame origin above the ground, per recording batch [m].

    Not documented and not constant: the origin sits 0.10 m above the ground in
    the earlier recordings and 0.07 m below it in the later ones.  Estimated as
    minus the median bottom-z of labelled pedestrians within 8 m.
    """
    bottoms = collections.defaultdict(list)
    for fr in frames:
        a = fr["annos"]
        near = (a["name"] == "Pedestrian") & (np.hypot(a["location"][:, 0], a["location"][:, 1]) < 8) \
            if len(a["name"]) else np.zeros(0, bool)
        bottoms[recording_of(fr)].extend(a["location"][near, 2])
    return {rec: -float(np.median(z)) if z else 0.0 for rec, z in bottoms.items()}


def _ego_T(frame):
    T = np.eye(4)
    T[:3, :3] = frame["ego2global_rotation"]
    T[:3, 3] = frame["ego2global_translation"]
    return T


def build_chains(frames):
    """Group frames into gap-free chains of consecutive clips.

    Returns a list of chains; a chain is a list of frame dicts in time order.
    """
    clips = collections.defaultdict(list)
    for fr in frames:
        clips[fr["seq_token"]].append(fr)
    pieces = []
    for clip in clips.values():
        clip.sort(key=lambda fr: fr["timestamp"])
        for k, fr in enumerate(clip):
            fr["clip_index"] = k  # position inside its clip; keeps scenario ids unique when a clip is cut
        ts = np.array([fr["timestamp"] for fr in clip]) / 1e6
        for part in np.split(np.arange(len(clip)), np.flatnonzero(np.diff(ts) > MAX_STEP_S) + 1):
            pieces.append([clip[i] for i in part])
    pieces.sort(key=lambda p: p[0]["timestamp"])
    chains = []
    for piece in pieces:
        prev = chains[-1][-1] if chains else None
        gap = (piece[0]["timestamp"] - prev["timestamp"]) / 1e6 if prev else np.inf
        if prev is not None and 0 < gap <= MAX_STEP_S and prev["map_token"] == piece[0]["map_token"]:
            chains[-1].extend(piece)
        else:
            chains.append(list(piece))
    # A few frames carry a wildly wrong ego pose; cut the chain around such jumps.
    out = []
    for chain in chains:
        xy = np.array([fr["ego2global_translation"][:2] for fr in chain])
        ts = np.array([fr["timestamp"] for fr in chain]) / 1e6
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(ts) if len(chain) > 1 else np.zeros(0)
        out += [[chain[i] for i in part] for part in np.split(np.arange(len(chain)), np.flatnonzero(speed > MAX_EGO_SPEED) + 1)]
    return out


def _link(live, objs, t):
    """Assign the objects of one frame to live tracks.

    live: {key: dict(name, t, xy, vel, has_vel, ident, fresh)} with the last sighting of each
    track; fresh means it was seen in the previous frame, has_vel that vel was measured.  objs: list of (name, xy, ident).
    Returns {object index: track key}.
    """
    keys = list(live)
    if not keys or not objs:
        return {}
    cost = np.full((len(keys), len(objs)), 1e6)
    for i, key in enumerate(keys):
        tr = live[key]
        dt = t - tr["t"]
        speed = float(np.linalg.norm(tr["vel"]))
        pred = tr["xy"] + tr["vel"] * dt
        for j, (name, xy, ident) in enumerate(objs):
            if name != tr["name"]:
                continue
            d_pred, d_last = np.linalg.norm(pred - xy), np.linalg.norm(tr["xy"] - xy)
            id_gate = _GATE[name] + _MAX_ACCEL[name] * dt * dt if tr["has_vel"] else _MAX_SPEED[name] * dt
            if ident is not None and ident == tr["ident"] and d_pred <= id_gate:
                cost[i, j] = _ID_BONUS * d_pred
            elif tr["fresh"] and d_pred <= _GATE[name] + 0.5 * speed * dt:
                cost[i, j] = d_pred
            elif not tr["fresh"] and d_last <= REACQUIRE_RADIUS:
                cost[i, j] = d_last + REACQUIRE_RADIUS
    rows, cols = linear_sum_assignment(cost)
    return {j: keys[i] for i, j in zip(rows, cols) if cost[i, j] < 1e6}


def chain_to_segment(chain, split, index, origin_heights=None):
    """Turn one chain into a segment in the layout of hnod.coda.load_segments.

    origin_heights: output of ego_origin_heights (computed from the chain if omitted).
    """
    origin_height = (origin_heights or ego_origin_heights(chain))[recording_of(chain[0])]
    F = len(chain)
    ts = np.array([fr["timestamp"] for fr in chain]) / 1e6
    ego_T = np.stack([_ego_T(fr) for fr in chain])
    ego_yaw = np.arctan2(ego_T[:, 1, 0], ego_T[:, 0, 0])
    tracks, live = {}, {}
    for i, fr in enumerate(chain):
        a = fr["annos"]
        n = len(a["name"])
        centre = a["location"] + np.c_[np.zeros((n, 2)), a["dimensions"][:, 2] / 2]
        world = centre @ ego_T[i, :3, :3].T + ego_T[i, :3, 3]
        id_count = collections.Counter(a["id"].tolist())
        idents = [(fr["seq_token"], int(k)) if id_count[int(k)] == 1 else None for k in a["id"]]
        for tr in live.values():
            tr["fresh"] = tr["i"] == i - 1
        match = _link(live, [(str(a["name"][j]), world[j, :2], idents[j]) for j in range(n)], ts[i])
        for j in range(n):
            name = str(a["name"][j])
            key = match.get(j)
            if key is None:
                key = f"{name}:{len(tracks)}"
                tracks[key] = dict(category=name, xyz=np.full((F, 3), np.nan), yaw=np.full(F, np.nan),
                                   lwh=np.full((F, 3), np.nan), occlusion=np.full(F, "", dtype=object),
                                   valid=np.zeros(F, bool))
                vel, has_vel = np.zeros(2), False
            else:
                prev = live[key]
                has_vel = prev["fresh"]
                vel = (world[j, :2] - prev["xy"]) / (ts[i] - prev["t"]) if has_vel else np.zeros(2)
            tr = tracks[key]
            w, l, h = a["dimensions"][j]
            tr["xyz"][i], tr["lwh"][i] = world[j], (l, w, h)
            tr["yaw"][i] = ego_yaw[i] + a["rotation_y"][j] + np.pi / 2
            tr["occlusion"][i], tr["valid"][i] = "Unknown", True
            live[key] = dict(name=name, i=i, t=ts[i], xy=world[j, :2].copy(), vel=vel, has_vel=has_vel,
                             ident=idents[j], fresh=True)
        live = {k: v for k, v in live.items() if i - v["i"] < MAX_MISSED}

    # The labels do not fix which end of a box is the front; use the direction of travel.
    for tr in tracks.values():
        idx = np.flatnonzero(tr["valid"])
        pair = idx[:-1][np.diff(idx) == 1]
        if len(pair) == 0:
            continue
        vel = (tr["xyz"][pair + 1, :2] - tr["xyz"][pair, :2]) / (ts[pair + 1] - ts[pair])[:, None]
        fast = np.linalg.norm(vel, axis=1) > _MOVING[tr["category"]]
        if fast.any():
            along = np.cos(tr["yaw"][pair[fast]] - np.arctan2(vel[fast, 1], vel[fast, 0]))
            if np.median(along) < 0:
                tr["yaw"] += np.pi

    return dict(dataset="robosense", sequence=str(chain[0]["seq_token"]), segment=index, split=split,
                rate_hz=RATE_HZ, frames=chain[0]["clip_index"] + np.arange(F), timestamps=ts, ego_T=ego_T,
                labelled=np.ones(F, bool),
                ego_size=EGO_SIZE, lidar_height=origin_height, stationary_radius=STATIONARY_RADIUS,
                tracks=tracks, lidar_paths=[fr["hs64_path"] for fr in chain],
                lidar_T=np.stack([fr["hs2global"] for fr in chain]), bev_keys=[fr["hs64_path"] for fr in chain],
                # Height above ground of the frame the sweep is stored in: the Hesai frame, except in the
                # last recording batch where sweeps are already in the ego frame (hs2livox is identity).
                lidar_heights=np.array([fr["hs2livox"][2, 3] for fr in chain]) + origin_height,
                seq_tokens=np.array([fr["seq_token"] for fr in chain]), map_token=chain[0]["map_token"])


def load_segments(pkl_dir, split, min_frames=1):
    frames = load_frames(pkl_dir, split)
    heights = ego_origin_heights(frames)
    return [chain_to_segment(c, split, k, heights) for k, c in enumerate(build_chains(frames)) if len(c) >= min_frames]
