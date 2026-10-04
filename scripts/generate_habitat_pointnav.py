#!/usr/bin/env python
"""Generate navigation scenarios in Habitat: sampled point-goal episodes, or VLN-CE / ObjectNav episode files.

Runs in the habitat-sim environment (python 3.9, see README).  For every scene it
draws random start/goal pairs, follows the navmesh shortest path at walking speed,
renders a forward RGB camera plus four depth cameras at every step, and then uses
exactly the same windowing / map / point-cloud code as the real datasets
(hnod.pipeline.segment_rows), so rows have the same schema as CODa, RoboSense and
JRDB: 10 past frames + current as images, 10 future steps as ground truth with
labelled points and a static map.  There are no other agents, so `future_tracks`
is empty.

    /venv/habitat/bin/python scripts/generate_habitat_pointnav.py --scenes /path/to/hm3d/train \\
        --out /dev/shm/hnod/hf_habitat --episodes-per-scene 40 --workers 8
"""
import argparse
import faulthandler
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import eval as hnod_eval  # noqa: E402
from hnod import lidar_bev, scenario  # noqa: E402
from hnod.io import FEATURES  # noqa: E402
from hnod.pipeline import camera_inputs, segment_rows, write_shards  # noqa: E402

RATE_HZ = 2.0            # steps per second, like coda_2hz
SPEED = 1.0              # m/s along the path
CAMERA_HEIGHT = 1.2      # m above the floor (a humanoid's head is higher than the robots' sensors)
HFOV = 90.0
WIDTH, HEIGHT = 640, 480
DEPTH_WIDTH, DEPTH_HEIGHT = 320, 240   # the four depth cameras; points are voxelised to 0.1 m anyway
MIN_PATH, MAX_PATH = 12.0, 40.0   # geodesic length of an episode [m]
STRIDE = 2               # steps between consecutive scenarios (1 s)
MAP_CONTEXT_FRAMES = 10  # +-5 s
MIN_STATIC_SPAN = 2      # 1 s: rendered depth has no moving objects, so persistence hardly matters
SHARD_ROWS = 128
EGO_SIZE = (0.6, 0.6, 1.7)  # a nominal humanoid
# Navmesh radius = the evaluator's agent radius.  A larger radius would keep the shortest path
# clear of the evaluator's collision boundary, but it also cuts furnished houses into pieces
# (0.4 m already blocks most passages between furniture); instead the path is planned at 0.3 m
# and scenarios whose path still collides under the evaluator (Recast's erosion is only accurate
# to a cell) are dropped.
NAV_RADIUS = 0.3
PATH_MARGIN = 0.15       # the path is pushed this far from the navmesh boundary (0.45 m from obstacles)
SIM_START_TIMEOUT = 900  # s; scene loading takes a few seconds to a few minutes
MIN_ISLAND_AREA = 15.0   # m^2; smaller navmesh islands are not worth an episode
# ego frame (x fwd, y left, z up) -> optical (x right, y down, z fwd)
_AXES = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)


def habitat_to_world(p):
    """Habitat (x right, y up, -z forward) -> a right-handed world with z up: (x, -z, y)."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def make_sim(scene, gpu, dataset_config=None):
    """Simulator for one scene: a .glb path, or a scene id of a scene dataset config (HSSD, HM3D).

    The navmesh is recomputed for the nominal humanoid (NAV_RADIUS, height 1.7 m) with the scene's
    static objects included, so that the recorded path keeps the evaluator's clearance.
    """
    import habitat_sim
    import magnum as mn
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene)
    if dataset_config:
        cfg.scene_dataset_config_file = str(dataset_config)
    cfg.enable_physics = False
    cfg.gpu_device_id = gpu
    cfg.override_scene_light_defaults = True
    cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    specs = []
    for uuid, typ, yaw in (("rgb", habitat_sim.SensorType.COLOR, 0.0), ("d0", habitat_sim.SensorType.DEPTH, 0.0),
                           ("d1", habitat_sim.SensorType.DEPTH, 90.0), ("d2", habitat_sim.SensorType.DEPTH, 180.0),
                           ("d3", habitat_sim.SensorType.DEPTH, 270.0)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.hfov = uuid, typ, HFOV
        s.resolution = [HEIGHT, WIDTH] if typ == habitat_sim.SensorType.COLOR else [DEPTH_HEIGHT, DEPTH_WIDTH]
        s.position = mn.Vector3(0, CAMERA_HEIGHT, 0)
        s.orientation = mn.Vector3(0, math.radians(yaw), 0)
        specs.append(s)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    # Several processes bringing up EGL contexts at the same instant occasionally crash the
    # driver; a file lock serialises simulator construction (scene loading stays parallel).
    with open("/tmp/habitat_sim.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Construction has hung in native code while holding this lock, stalling every other scene;
        # the default SIGALRM action ends the process (and frees the lock) even then.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(SIM_START_TIMEOUT)
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        signal.alarm(0)
    nav = habitat_sim.NavMeshSettings()
    nav.set_defaults()
    nav.agent_radius, nav.agent_height = NAV_RADIUS, EGO_SIZE[2]
    nav.include_static_objects = True
    sim.recompute_navmesh(sim.pathfinder, nav)
    return sim


def intrinsics(width=WIDTH, height=HEIGHT):
    f = width / 2 / math.tan(math.radians(HFOV) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def depth_to_points(depth, yaw_deg, K):
    """Depth image of a camera yawed by yaw_deg about the up axis -> points in the ego frame."""
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = depth.astype(np.float64)
    ok = (z > 0.05) & (z < 25.0)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x[ok], y[ok], z[ok]], 1)
    ego = cam @ _AXES  # optical -> ego (inverse of A, which is orthonormal): p_ego = A^T p_cam
    ego[:, 2] += CAMERA_HEIGHT
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = np.array([[c, -s], [s, c]])
    ego[:, :2] = ego[:, :2] @ rot.T
    return ego


def choose_island(sim):
    """Index of the navmesh island to walk on: the enclosed one.

    A recomputed navmesh also covers roofs and outdoor ground.  Among the large islands the one
    whose horizontal depth cameras see the least sky (no-hit pixels) is the interior.
    """
    from habitat_sim.utils.common import quat_from_angle_axis
    pf, agent = sim.pathfinder, sim.get_agent(0)
    areas = np.array([pf.island_area(i) for i in range(pf.num_islands)])
    best, best_sky = 0, np.inf
    for i in np.flatnonzero(areas >= min(MIN_ISLAND_AREA, areas.max())):
        sky = []
        for _ in range(8):
            st = agent.get_state()
            st.position = pf.get_random_navigable_point(island_index=int(i))
            st.rotation = quat_from_angle_axis(0.0, np.array([0, 1.0, 0]))
            agent.set_state(st)
            obs = sim.get_sensor_observations()
            sky.append(np.mean([(obs[f"d{k}"] <= 0).mean() for k in range(4)]))
        if np.mean(sky) < best_sky - 1e-6 or (abs(np.mean(sky) - best_sky) <= 1e-6 and areas[i] > areas[best]):
            best, best_sky = int(i), float(np.mean(sky))
    return best


def load_regions(dataset_config, scene):
    """Room polygons of a scene (HSSD semantic_config.json): [(matplotlib Path in (x, z), floor_y)]."""
    if not dataset_config:
        return None
    f = Path(dataset_config).parent / "semantics/scenes" / f"{scene}.semantic_config.json"
    if not f.exists():
        return None
    from matplotlib.path import Path as MplPath
    out = []
    for r in json.load(open(f))["region_annotations"]:
        loop = np.array(r["poly_loop"], dtype=np.float64)
        if len(loop) >= 3:
            out.append((MplPath(loop[:, [0, 2]]), float(r["floor_height"])))
    return out or None


def indoor(points, regions):
    """Which habitat points lie inside a room polygon at its floor height."""
    points = np.atleast_2d(points)
    ok = np.zeros(len(points), bool)
    for poly, floor in regions:
        # the navmesh lies up to ~0.7 m above the annotated floor height (thick floors, rugs)
        ok |= poly.contains_points(points[:, [0, 2]]) & (points[:, 1] > floor - 0.3) & (points[:, 1] < floor + 1.2)
    return ok


def sample_episode(pf, rng, island, regions=None):
    """A navmesh shortest path MIN_PATH..MAX_PATH long: on the given island, or (island None) between
    any two indoor navigable points when room polygons are known."""
    import habitat_sim
    for _ in range(300 if island is not None else 3000):
        if island is not None:
            a, b = pf.get_random_navigable_point(island_index=island), pf.get_random_navigable_point(island_index=island)
        else:
            a, b = pf.get_random_navigable_point(), pf.get_random_navigable_point()
            if not indoor([a, b], regions).all():
                continue
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = a, b
        if pf.find_path(path) and MIN_PATH <= path.geodesic_distance <= MAX_PATH and len(path.points) >= 2:
            pts = clear_path(pf, np.array(path.points, dtype=np.float64))
            if pts is None or (regions is not None and indoor(pts, regions).mean() < 0.98):
                continue
            return pts
    return None


def clear_path(pf, points, margin=PATH_MARGIN, iters=8):
    """Push a navmesh shortest path away from the navmesh boundary and round its corners.

    The shortest path hugs obstacles at exactly the navmesh radius, which is also the evaluator's
    agent radius; a walker keeps more room.  Points closer than `margin` to the boundary are
    moved along the obstacle normal, then the polyline is smoothed; both only where the result
    is still navigable.  Returns a polyline sampled every 0.25 m, or None if a segment left
    the navmesh.
    """
    pts = resample_path(points, 0.25)[0]
    for _ in range(iters):
        new = pts.copy()
        for i in range(len(pts)):
            hit = pf.closest_obstacle_surface_point(pts[i], margin + 0.5)
            if hit.hit_dist < margin:
                q = pts[i] + np.asarray(hit.hit_normal, dtype=np.float64) * (margin - hit.hit_dist)
                if pf.is_navigable(q):
                    new[i] = q
        sm = new.copy()
        sm[1:-1] = 0.25 * new[:-2] + 0.5 * new[1:-1] + 0.25 * new[2:]
        for i in range(1, len(sm) - 1):
            if not pf.is_navigable(sm[i]):
                sm[i] = new[i]
        pts = sm
    mid = 0.5 * (pts[:-1] + pts[1:])
    if not all(pf.is_navigable(m) for m in mid):
        return None
    return pts


def load_episodes(files, task, scene_key=None):
    """Episodes of a habitat-lab dataset file, grouped by scene: {scene_id: [episode, ...]}.

    task "vln": VLN-CE R2R / RxR files ({"episodes": [{scene_id, start_position, reference_path,
    instruction: {instruction_text}}]}); the path visits the reference waypoints.
    task "objectnav": ObjectNav content files ({"episodes": [{scene_id, start_position,
    object_category}], "goals_by_category": {"<scene file>_<category>": [{view_points}]}}); the path
    is the shortest one to any view point of any object of the category.
    Each returned episode is dict(id, start, waypoints | goals, instruction).
    """
    import gzip
    out = {}
    for f in files:
        d = json.load(gzip.open(f) if str(f).endswith(".gz") else open(f))
        views = {}
        for ep in d.get("episodes", []):
            sid = ep["scene_id"]
            if scene_key is not None and sid != scene_key:
                continue
            e = dict(id=str(ep["episode_id"]), start=ep["start_position"])
            if task == "vln":
                e["waypoints"] = ep["reference_path"]
                e["instruction"] = ep["instruction"]["instruction_text"].strip()
            elif task == "objectnav":
                cat = ep["object_category"]
                key = f"{Path(sid).name}_{cat}"
                if key not in views:
                    views[key] = [v["agent_state"]["position"] for g in d["goals_by_category"].get(key, [])
                                  for v in g.get("view_points", [])]
                if not views[key]:
                    continue
                e["goals"] = views[key]
                e["instruction"] = f"Find a {cat.replace('_', ' ')} and stop near it."
            else:
                raise ValueError(task)
            out.setdefault(sid, []).append(e)
    return out


def episode_path(pf, ep):
    """Navmesh path of a dataset episode (habitat coordinates), or None if it cannot be followed."""
    import habitat_sim
    start = pf.snap_point(np.asarray(ep["start"], dtype=np.float32))
    if not np.isfinite(np.asarray(start)).all():
        return None
    if "goals" in ep:
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start
        path.requested_ends = [np.asarray(g, dtype=np.float32) for g in ep["goals"]]
        if not pf.find_path(path) or len(path.points) < 2:
            return None
        pts = np.array(path.points, dtype=np.float64)
    else:
        legs, prev = [], start
        for w in ep["waypoints"][1:]:
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = prev, pf.snap_point(np.asarray(w, dtype=np.float32))
            if not pf.find_path(path) or len(path.points) < 2:
                return None
            legs.append(np.array(path.points, dtype=np.float64)[(1 if legs else 0):])
            prev = path.requested_end
        if not legs:
            return None
        pts = np.concatenate(legs)
    if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < 1.0:
        return None
    return clear_path(pf, pts)


def pad_standing(pos, yaw):
    """An episode starts from rest and ends with a stop: stand for the past / future window."""
    n0, n1 = scenario.N_PAST, scenario.N_FUTURE
    return (np.concatenate([np.repeat(pos[:1], n0, 0), pos, np.repeat(pos[-1:], n1, 0)]),
            np.concatenate([np.repeat(yaw[:1], n0), yaw, np.repeat(yaw[-1:], n1)]))


def resample_path(points, step):
    """Polyline -> positions every `step` metres plus the heading of travel (habitat coords)."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    dist = np.arange(0, cum[-1], step)
    pos = np.stack([np.interp(dist, cum, points[:, k]) for k in range(3)], 1)
    # heading from the smoothed direction of travel (habitat: forward is -z)
    d = np.gradient(pos, axis=0)
    yaw = np.arctan2(-d[:, 0], -d[:, 2])  # angle about +y so that yaw=0 faces -z
    return pos, yaw


def episode_segment(sim, pos, yaw, images_dir, name, rng, dataset="habitat", task="pointgoal", instruction=""):
    """Follow a path step by step; return a segment dict plus packed points and grids per step."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    K, Kd = intrinsics(), intrinsics(DEPTH_WIDTH, DEPTH_HEIGHT)
    agent = sim.get_agent(0)
    F = len(pos)
    ego_T = np.tile(np.eye(4), (F, 1, 1))
    store, grids = {}, {}
    (Path(images_dir) / name).mkdir(parents=True, exist_ok=True)
    for i in range(F):
        st = agent.get_state()
        st.position = pos[i]
        st.rotation = quat_from_angle_axis(float(yaw[i]), np.array([0, 1.0, 0]))
        agent.set_state(st)
        if i == 0 or not (np.array_equal(pos[i], pos[i - 1]) and yaw[i] == yaw[i - 1]):
            obs = sim.get_sensor_observations()  # a standing agent sees the same thing again
        # ego pose in the z-up world: position on the floor, yaw about z.  A habitat yaw of t about +y
        # points the agent along (-sin t, 0, -cos t), i.e. along (-sin t, cos t) in the (x, -z) world,
        # whose heading angle is t + pi/2.
        c, s = math.cos(yaw[i] + math.pi / 2), math.sin(yaw[i] + math.pi / 2)
        ego_T[i, :3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        ego_T[i, :3, 3] = habitat_to_world(pos[i])
        cv2.imwrite(str(Path(images_dir) / name / f"{i:04d}.jpg"), cv2.cvtColor(obs["rgb"][:, :, :3], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        pts = np.concatenate([depth_to_points(obs[f"d{k}"], 90.0 * k, Kd) for k in range(4)])
        h = pts[:, 2]  # height above the floor: the agent stands on the navmesh floor
        keep = np.linalg.norm(pts[:, :2], axis=1) > 0.3
        pts, h = pts[keep], h[keep]
        # world-aligned grid cell of every point (lidar_bev conventions) for the static map
        world = pts @ ego_T[i, :3, :3].T + ego_T[i, :3, 3]
        origin = np.floor(ego_T[i, :2, 3] / lidar_bev.RES).astype(np.int64) - lidar_bev.FRAME_GRID // 2
        fi = np.floor(world[:, 0] / lidar_bev.RES).astype(np.int64) - origin[0]
        fj = np.floor(world[:, 1] / lidar_bev.RES).astype(np.int64) - origin[1]
        inside = (fi >= 0) & (fi < lidar_bev.FRAME_GRID) & (fj >= 0) & (fj < lidar_bev.FRAME_GRID)
        grid = lidar_bev.rasterize(fi[inside], fj[inside], h[inside])
        grids[(name, i)] = ((int(origin[0]), int(origin[1])), zlib.compress(grid.tobytes(), 6))
        store[(name, i)] = lidar_bev.pack_points(*lidar_bev.downsample_points(pts, h))
    ts = np.arange(F) / RATE_HZ
    camera = dict(name="rgb", width=WIDTH, height=HEIGHT, K=K, T_camera_from_ego=np.block(
        [[_AXES, (-_AXES @ np.array([0, 0, CAMERA_HEIGHT]))[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    seg = dict(dataset=dataset, sequence=name, segment=0, rate_hz=RATE_HZ, frames=np.arange(F), timestamps=ts,
               ego_T=ego_T, labelled=np.ones(F, bool), ego_size=EGO_SIZE, lidar_height=0.0, tracks={}, camera=camera,
               point_keys=[(name, i) for i in range(F)], bev_keys=[(name, i) for i in range(F)],
               image_paths=[str(Path(images_dir) / name / f"{i:04d}.jpg") for i in range(F)],
               task=task, instruction=instruction)
    return seg, store, grids


def work(args):
    scene, n_episodes, seed, images_dir, out, gpu, job, dataset_config, dataset = args[:9]
    spec = args[9] if len(args) > 9 else None   # episode-driven: dict(task, files, scene_id)
    done = Path(out) / f"job{job:05d}.done"
    if done.exists():
        return job, int(done.read_text())
    rng = np.random.default_rng(seed)
    sim = make_sim(scene, gpu, dataset_config)
    rows = []
    stats = dict(no_path=0, short=0, candidates=0)
    try:
        pf = sim.pathfinder
        if not pf.is_loaded:
            return job, 0
        task, episodes = "pointgoal", [None] * n_episodes
        if spec:
            task = spec["task"]
            episodes = load_episodes(spec["files"], task, spec["scene_id"]).get(spec["scene_id"], [])
            episodes = [episodes[i] for i in rng.permutation(len(episodes))[:n_episodes]]
            stats["area"] = float(pf.navigable_area)
        else:
            regions = load_regions(dataset_config, scene)
            # With room polygons, episodes are drawn between indoor points on any island (upper floors
            # and the outdoors are separate islands); without them, the enclosed island is guessed.
            island = None if regions else choose_island(sim)
            stats["area"] = pf.island_area(island) if island is not None else float(pf.navigable_area)
        for e, ep in enumerate(episodes):
            path = episode_path(pf, ep) if ep else sample_episode(pf, rng, island, regions)
            if path is None:
                stats["no_path"] += 1
                continue
            pos, yaw = resample_path(path, SPEED / RATE_HZ)
            if ep:
                pos, yaw = pad_standing(pos, yaw)
            if len(pos) < scenario.N_STEPS + 2:
                stats["short"] += 1
                continue
            name = f"{Path(scene).stem.split('.')[0]}_{ep['id'] if ep else seed}_{e:03d}"
            seg, store, grids = episode_segment(sim, pos, yaw, images_dir, name, rng, dataset, task,
                                                ep["instruction"] if ep else "")
            scenario.clean_segment(seg, lambda c: scenario.STATIC)
            new = segment_rows(seg, grids, 1, STRIDE, MAP_CONTEXT_FRAMES, MIN_STATIC_SPAN, 5, points=store,
                               camera=camera_inputs)
            # The planned path may graze an obstacle at the evaluator's radius (doorways leave no
            # room for a margin); keep only rows whose path is clean.  Evaluated on the values as
            # stored (ego in float32): map distances are quantised, so a tie must not flip on reload.
            stats["candidates"] += len(new)
            for row in new:
                stored = dict(row, ego={k: (np.asarray(v, np.float32).tolist() if np.ndim(v) else v)
                                        for k, v in row["ego"].items()})
                r = hnod_eval.evaluate_scenario(stored, hnod_eval.baseline_expert(stored))
                if not (r["collided"] or r["collided_lidar"]):
                    rows.append(row)
            shutil.rmtree(Path(images_dir) / name, ignore_errors=True)
    finally:
        sim.close()
    write_shards(rows, out, f"job{job:05d}", FEATURES, SHARD_ROWS)
    done.write_text(str(len(rows)))
    print(f"{Path(scene).name}: {len(rows)} scenarios kept of {stats['candidates']} ({stats['no_path']} episodes without "
          f"a path, {stats['short']} too short, island area {stats.get('area', 0):.0f} m2)", flush=True)
    return job, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", help="directory searched recursively for *.glb scenes with a .navmesh")
    ap.add_argument("--dataset-config", help="a habitat *.scene_dataset_config.json (HSSD, HM3D); all its scenes are used")
    ap.add_argument("--splits", help="yaml with scene ids under train / val: val scenes become validation + test")
    ap.add_argument("--config", default="habitat_2hz", help="name of the dataset config (output sub-directory)")
    ap.add_argument("--episodes", nargs="*", help="habitat-lab episode files (VLN-CE or ObjectNav *.json.gz) to follow "
                                                  "instead of sampling point-goal episodes")
    ap.add_argument("--task", choices=["vln", "objectnav"], help="format of --episodes")
    ap.add_argument("--scene-root", help="directory that the episodes' scene_id paths are relative to")
    ap.add_argument("--split", default="train", help="split that --episodes belong to")
    ap.add_argument("--out", help="output directory (required unless --job)")
    ap.add_argument("--images-dir", default="/dev/shm/hnod/habitat_frames")
    ap.add_argument("--episodes-per-scene", type=int, default=40)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.1, help="fraction of scenes held out as validation")
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="render with Mesa llvmpipe instead of the GPU (see tools/egl_software_only.c)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-timeout", type=float, default=4 * 3600, help="seconds per scene")
    ap.add_argument("--job", help=argparse.SUPPRESS)  # internal: run one job (JSON) in this process
    args = ap.parse_args()
    if args.job:
        faulthandler.enable(all_threads=True)
        work(tuple(json.loads(args.job)))
        return
    if not args.out:
        ap.error("--out is required")
    if args.cpu:
        shim = Path(__file__).resolve().parents[1] / "tools/egl_software_only.so"
        if not shim.exists():
            raise SystemExit(f"build {shim} first: gcc -shared -fPIC -O2 -o {shim} {shim.with_suffix('.c')} -ldl")
        os.environ.update(LD_PRELOAD=str(shim), LIBGL_ALWAYS_SOFTWARE="1",
                          __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        args.gpus = 1
    by_scene = None
    if args.episodes:
        if not (args.task and args.scene_root):
            ap.error("--episodes needs --task and --scene-root")
        by_scene = {}
        for f in args.episodes:   # which files mention which scene; the jobs read the episodes themselves
            for sid in load_episodes([f], args.task):
                by_scene.setdefault(sid, []).append(str(f))
        scenes = sorted(sid for sid in by_scene if (Path(args.scene_root) / sid).exists())
        print(f"{len(by_scene)} scenes in the episode files, {len(scenes)} found under {args.scene_root}", flush=True)
    elif args.dataset_config:
        cfg = json.load(open(args.dataset_config))
        root = Path(args.dataset_config).parent
        scenes = sorted(p.name.split(".")[0] for d in cfg["scene_instances"]["paths"][".json"]
                        for p in (root / d).glob("*.scene_instance.json"))
    else:
        scenes = sorted(str(p) for p in Path(args.scenes).rglob("*.glb") if p.with_suffix(".navmesh").exists()
                        or (p.parent / (p.stem + ".basis.navmesh")).exists())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    rng = np.random.default_rng(args.seed)
    split_of = {}
    if by_scene is not None:
        split_of = {s: args.split for s in scenes}
    elif args.splits:
        import yaml
        held = set(yaml.safe_load(open(args.splits)).get("val", []))
        held = [s for s in scenes if Path(s).name.split(".")[0] in held]
        order = rng.permutation(len(held))
        n_val = int(round(len(held) * args.val_fraction / (args.val_fraction + args.test_fraction)))
        for k, i in enumerate(order):
            split_of[held[i]] = "validation" if k < n_val else "test"
        for s in scenes:
            split_of.setdefault(s, "train")
    else:
        order = rng.permutation(len(scenes))
        n_val, n_test = int(len(scenes) * args.val_fraction), int(len(scenes) * args.test_fraction)
        for k, i in enumerate(order):
            split_of[scenes[i]] = "validation" if k < n_val else "test" if k < n_val + n_test else "train"
    counts = {sp: sum(v == sp for v in split_of.values()) for sp in ("train", "validation", "test")}
    print(f"{len(scenes)} scenes: {counts}", flush=True)
    out = Path(args.out) / args.config
    jobs = []
    for j, scene in enumerate(scenes):
        parts = out / "_parts" / split_of[scene]
        parts.mkdir(parents=True, exist_ok=True)
        job = (str(scene), args.episodes_per_scene, args.seed * 100000 + j, args.images_dir, str(parts),
               -1 if args.cpu else j % args.gpus, j, args.dataset_config, args.config.split("_")[0])
        if by_scene is not None:
            job = (str(Path(args.scene_root) / scene),) + job[1:] + (dict(task=args.task, files=by_scene[scene], scene_id=scene),)
        jobs.append(job)
    # One subprocess per scene: habitat-sim occasionally dies in native code, and a forked
    # process cannot bring up its own EGL context; a crash then costs one scene, not the run.
    logs = out / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(job):
        done = Path(job[4]) / f"job{job[6]:05d}.done"
        if done.exists():
            return job[6], int(done.read_text()), "cached"
        with open(logs / f"job{job[6]:05d}.log", "w") as log:
            try:
                p = subprocess.run([sys.executable, __file__, "--job", json.dumps(job)], stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.job_timeout)
                status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
        n = int(done.read_text()) if done.exists() else 0
        return job[6], n, status

    with ThreadPoolExecutor(args.workers) as ex:
        for k, (job, n, status) in enumerate(ex.map(run, jobs)):
            line = [ln for ln in open(logs / f"job{job:05d}.log", errors="replace") if " scenarios kept" in ln] \
                if (logs / f"job{job:05d}.log").exists() else []
            print(f"[{k + 1}/{len(jobs)}] job {job} {status}: " + (line[-1].strip() if line else f"{n} scenarios"), flush=True)
    total = 0
    for split in ("train", "validation", "test"):
        total += sum(int(f.read_text()) for f in (out / "_parts" / split).glob("*.done"))
        parts = sorted((out / "_parts" / split).glob("*.parquet"))
        for f in parts:   # same names as scripts/sync_parts_hf.py uploads while the run is going
            dest = out / f"{split}-{f.name}"
            if not dest.exists():
                os.link(f, dest)
        print(f"{args.config} {split:10s} {len(parts)} shards")
    # _parts (shards + .done markers) is kept so that the run can be resumed or extended.
    json.dump({str(k): v for k, v in split_of.items()}, open(out / "scene_splits.json", "w"), indent=1)
    print(f"{total} scenarios")


if __name__ == "__main__":
    main()
