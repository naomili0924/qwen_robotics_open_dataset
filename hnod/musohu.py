"""MuSoHu (Multi-Modal Social Human Navigation, CC0) reader: one ROS1 bag -> segments.

A person walks with a helmet carrying a Velodyne VLP-16 (frame `lidar_link` = `base_link`) and a ZED 2
(left image `zed2/zed_node/rgb/image_rect_color/compressed`, 1280 x 720, rectified; odometry
`zed2/zed_node/odom`, `odom` -> `base_link`).  There are no object labels: moving people are found
in the lidar by hnod.lidar_tracks.

The helmet pitches and rolls with the head.  Every sweep is therefore levelled with the odometry's
roll and pitch before anything else, and the per-frame pose handed to the shared pipeline is
yaw-only (ego_T: gravity-aligned position + heading of the walker).  Ground estimation, maps and
published points then work exactly as for a wheeled robot.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

LIDAR_TOPIC = "/velodyne_points"
ODOM_TOPIC = "/zed2/zed_node/odom"
IMAGE_TOPIC = "/zed2/zed_node/rgb/image_rect_color/compressed"
INFO_TOPIC = "/zed2/zed_node/rgb/camera_info"
RATE_HZ = 10.0
EGO_SIZE = (0.5, 0.6, 1.75)   # a walking person: length, width, height [m]
MAX_GAP_S = 0.5               # odometry or lidar gap that cuts a segment
IMAGE_MATCH_S = 0.05          # an image belongs to a sweep if it is this close in time
MIN_SEGMENT_FRAMES = 150      # 15 s

# base_link -> zed2_left_camera_optical_frame, from the bags' /tf_static
_q = lambda x, y, z, w: Rotation.from_quat([x, y, z, w]).as_matrix()  # noqa: E731


def _T(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


BASE_FROM_OPTICAL = (_T(t=(0.16, 0, 0)) @ _T(_q(0, 0.025, 0, 1), (0, 0, 0.015)) @ _T(t=(0, 0.06, 0))
                     @ _T(_q(0.5, -0.5, 0.5, -0.5), (-0.01, 0, 0)))


def _cloud_xyz(msg):
    """x, y, z float32 of a PointCloud2 (any point_step)."""
    off = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(n, msg.point_step)
    xyz = np.stack([buf[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
    return xyz[np.isfinite(xyz).all(1)]


def read_bag(path, image_dir):
    """Sweeps, poses and images of one bag.

    Writes the images that match a sweep to image_dir/<frame>.jpg.  Returns dict(
    t (F,) sweep times [s], sweeps [F x (N, 3) lidar-frame points], R (F, 3, 3) / p (F, 3) pose of
    base_link in `odom` at the sweep times, image_paths [F, path or None], camera dict).
    """
    from rosbags.highlevel import AnyReader
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    ot, opos, oquat, sweeps, st, imgs, camera = [], [], [], [], [], [], None
    with AnyReader([Path(path)]) as r:
        conns = [c for c in r.connections if c.topic in (LIDAR_TOPIC, IMAGE_TOPIC, INFO_TOPIC)
                 or (c.topic == ODOM_TOPIC and getattr(c.ext, "callerid", "") in ("/zed2/zed_node", None))]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if c.topic == ODOM_TOPIC:
                p, q = m.pose.pose.position, m.pose.pose.orientation
                ot.append(t)
                opos.append((p.x, p.y, p.z))
                oquat.append((q.x, q.y, q.z, q.w))
            elif c.topic == LIDAR_TOPIC:
                st.append(t)
                sweeps.append(_cloud_xyz(m))
            elif c.topic == IMAGE_TOPIC:
                imgs.append((t, bytes(m.data)))
            elif camera is None:
                camera = dict(name="zed2_left_rect", width=int(m.width), height=int(m.height),
                              K=np.asarray(m.K, float).reshape(3, 3),
                              T_camera_from_ego=np.linalg.inv(BASE_FROM_OPTICAL))
    ot, opos, oquat = np.asarray(ot), np.asarray(opos), np.asarray(oquat)
    order = np.argsort(ot)
    ot, opos, oquat = ot[order], opos[order], oquat[order]
    _, uniq = np.unique(ot, return_index=True)
    ot, opos, oquat = ot[uniq], opos[uniq], oquat[uniq]
    st = np.asarray(st)
    ok = (st >= ot[0]) & (st <= ot[-1])
    st, sweeps = st[ok], [s for s, k in zip(sweeps, ok) if k]
    slerp = Slerp(ot, Rotation.from_quat(oquat))
    R = slerp(st).as_matrix()
    p = np.stack([np.interp(st, ot, opos[:, k]) for k in range(3)], 1)
    # an odometry gap around a sweep makes its pose unreliable
    gap = np.diff(ot)[np.clip(np.searchsorted(ot, st) - 1, 0, len(ot) - 2)]
    it = np.array([t for t, _ in imgs])
    paths = []
    for i, t in enumerate(st):
        k = int(np.argmin(np.abs(it - t))) if len(it) else -1
        if k >= 0 and abs(it[k] - t) <= IMAGE_MATCH_S:
            f = image_dir / f"{i:06d}.jpg"
            f.write_bytes(imgs[k][1])
            paths.append(str(f))
        else:
            paths.append(None)
    return dict(t=st, sweeps=sweeps, R=R, p=p, odom_gap=gap, image_paths=paths, camera=camera)


def level(R):
    """(F, 3, 3) roll-pitch part of each pose (yaw removed) and the yaw angles."""
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.zeros_like(R)
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1
    return np.einsum("fji,fjk->fik", Rz, R), yaw  # Rz^T R


def floor_plane(pts, iters=150, tol=0.05, min_inliers=300, seed=0):
    """Floor plane of one sweep: (unit normal pointing up, height of the sensor above it) or None.

    RANSAC over the low returns 3-15 m away (the floor; a VLP-16 at head height sees it only there),
    refined by least squares on the inliers.
    """
    rng = np.random.default_rng(seed)
    r = np.hypot(pts[:, 0], pts[:, 1])
    c = pts[(r > 3.0) & (r < 15.0) & (pts[:, 2] < -0.8)].astype(np.float64)
    if len(c) < min_inliers:
        return None
    if len(c) > 4000:
        c = c[rng.choice(len(c), 4000, replace=False)]
    tri = c[rng.integers(0, len(c), (iters, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    k = np.linalg.norm(n, axis=1)
    good = k > 1e-6
    n = n[good] / k[good, None]
    n *= np.where(n[:, 2] < 0, -1.0, 1.0)[:, None]
    keep = n[:, 2] >= 0.9  # more than ~25 degrees off vertical: not the floor
    n, origin = n[keep], tri[good][keep, 0]
    if not len(n):
        return None
    counts = (np.abs(c @ n.T - (origin * n).sum(1)) < tol).sum(0)
    b = int(np.argmax(counts))
    if counts[b] < min_inliers:
        return None
    q = c[np.abs((c - origin[b]) @ n[b]) < tol]
    centre = q.mean(0)
    n = np.linalg.svd(q - centre)[2][2]
    n = n * np.sign(n[2])
    return n, float(-(centre @ n))


def rotation_to_up(n):
    """Rotation taking unit vector n to +z (the smallest one)."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def level_by_floor(sweeps, R_odom, smooth=5):
    """Per sweep the levelling rotation (sensor -> gravity-aligned, yaw untouched) and the sensor height.

    Uses the floor seen by the lidar; sweeps where it is not seen take the odometry's tilt.  Normals
    are median-filtered over +-smooth sweeps (heads move, but not at 10 Hz).
    """
    F = len(sweeps)
    Rl_odom, _ = level(R_odom)
    normals = np.full((F, 3), np.nan)
    heights = np.full(F, np.nan)
    for i, pts in enumerate(sweeps):
        fit = floor_plane(pts, seed=i)
        if fit is not None:
            normals[i], heights[i] = fit
    ok = np.isfinite(normals).all(1)
    odom_n = Rl_odom.transpose(0, 2, 1)[:, :, 2]  # the odometry's up vector in the sensor frame
    normals[~ok] = odom_n[~ok]
    pad = np.pad(normals, ((smooth, smooth), (0, 0)), mode="edge")
    sm = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * smooth + 1, axis=0), axis=2)
    sm /= np.linalg.norm(sm, axis=1, keepdims=True)
    R = np.stack([rotation_to_up(n) for n in sm])
    height = float(np.nanmedian(heights)) if ok.any() else 1.7
    return R, height, float(ok.mean())


def cut_segments(t, odom_gap, min_frames=MIN_SEGMENT_FRAMES):
    brk = np.r_[False, np.diff(t) > MAX_GAP_S] | (odom_gap > MAX_GAP_S)
    starts = np.flatnonzero(np.r_[True, brk[1:]])
    ends = np.r_[starts[1:], len(t)]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_frames]
