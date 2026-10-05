import numpy as np
from datasets import Dataset

from hnod import frames as fr
from hnod.windows import FrameWindows, WindowConfig, to_frame


def _episode(eid, n=60, speed=0.5, rate=5.0, heading=0.3, segments=(), at_rest=False):
    t = np.arange(n) / rate
    d = speed * t
    x, y = 1.0 + d * np.cos(heading), -2.0 + d * np.sin(heading)
    img = {"bytes": fr.jpeg_bytes(np.zeros((8, 8, 3), np.uint8)), "path": None}
    rows = [dict(episode_id=eid, frame_index=i, source_frame=i, timestamp=float(t[i]), image=img, image_right=None,
                 pose=[float(x[i]), float(y[i]), 0.0, heading]) for i in range(n)]
    ep = dict(episode_id=eid, dataset="synthetic", source_sequence=eid, split="train", num_frames=n, rate_hz=rate,
              duration_s=float(t[-1]), path_length_m=float(d[-1]), median_speed_mps=speed,
              embodiment="person_walking", environment="indoor", environment_method="labelled", indoor_fraction=1.0,
              frame_indoor_prob=[1.0] * n,
              camera=dict(name="front", width=8, height=8, K=[1, 0, 4, 0, 1, 4, 0, 0, 1], distortion_model="none",
                          distortion=[], height_m=1.3, stereo_baseline_m=float("nan")),
              segments=list(segments), ends_at_rest=at_rest, fits="both", licence="test")
    return rows, ep


def _windows(cfg=None, **kw):
    r1, e1 = _episode("a", **kw)
    r2, e2 = _episode("b", n=40, heading=-1.0)
    frames = Dataset.from_list(r1 + r2, features=fr.FRAME_FEATURES)
    eps = Dataset.from_list([e1, e2], features=fr.EPISODE_FEATURES)
    return FrameWindows(frames, eps, cfg or WindowConfig(n_past=3, past_dt_s=0.4))


def test_cut_episodes_at_nulls_and_jumps():
    t = np.arange(100) * 0.2
    xy = np.c_[0.1 * np.arange(100), np.zeros(100)]
    valid = np.ones(100, bool)
    valid[40] = False
    xy[70:] += 11.0  # odometry re-initialisation
    assert fr.cut_episodes(t, xy, valid, min_frames=5, min_length_m=0.5) == [(0, 40), (41, 70), (70, 100)]


def test_yaw_from_quat():
    a = 0.7
    assert np.isclose(fr.yaw_from_quat(0, 0, np.sin(a / 2), np.cos(a / 2)), a)


def test_waypoints_by_distance_in_robot_frame():
    w = _windows()
    s = w.sample(0)
    # straight walk along the robot's heading: waypoints on +x, 0.25 m apart, zero relative yaw
    assert np.allclose(s["target"][:, 0], 0.25 * np.arange(1, 9), atol=1e-6)
    assert np.allclose(s["target"][:, 1:], 0, atol=1e-6)
    assert not s["stop"].any()
    # history clamped to the episode start, masked
    assert s["history_mask"].sum() == 1 and (s["history_rows"] == s["row"]).all()
    # goal beyond the horizon, straight ahead
    assert s["goal"][0] > 8 * 0.25 and abs(s["goal"][1]) < 1e-6
    assert "(" in s["prompt"] and s["task"] == "pointgoal"


def test_no_stop_samples_unless_episode_ends_at_rest():
    w = _windows()
    for k in range(len(w)):
        assert not w.sample(k)["stop"].any()
    w = _windows(at_rest=True)
    assert any(w.sample(k)["stop"].any() for k in range(len(w)))


def test_language_segment_goal_and_prompt():
    seg = dict(start_frame=0, end_frame=50, task="language", instruction="A white door.", brief="door",
               source="test")
    w = _windows(WindowConfig(n_past=3, language_fraction=1.0, language_point_fraction=0.0), segments=[seg])
    s = w.sample(0)
    assert s["task"] == "language" and "a white door" in s["prompt"] and "(" not in s["prompt"]
    assert np.isclose(s["goal"][0], 0.5 * 50 / 5.0, atol=1e-5)  # pose at end_frame, 5 m ahead


def test_to_frame_rotation():
    p0 = np.array([1.0, 1.0, 0.0, np.pi / 2])
    assert np.allclose(to_frame(p0, [1.0, 2.0]), [1.0, 0.0])
    assert np.allclose(to_frame(p0, [0.0, 1.0]), [0.0, 1.0])


def test_images_follow_rows():
    w = _windows()
    item = w[5]
    assert len(item["images"]) == 4


def test_standing_jitter_is_not_travel():
    from hnod.windows import arc_length
    rng = np.random.default_rng(0)
    still = rng.normal(0, 0.01, (200, 2))
    walk = np.c_[np.linspace(0, 5, 50), np.zeros(50)] + still[-1]
    s = arc_length(np.r_[still, walk])
    assert s[199] < 0.1 and abs(s[-1] - 5) < 0.2


def test_frame_nav_dataset_items_match_the_scenario_layout():
    from vla.config import Config
    from vla.data import KIN_DIM, FrameNavDataset, fit_action_scale_frames, window_config
    cfg = Config(data_format="frames", horizon=8, frames=4, action_scale=2.0)
    ds = FrameNavDataset([_windows(window_config(cfg)), _windows(window_config(cfg))], cfg)
    item = ds[3]
    assert item["kin"].shape == (KIN_DIM,) and item["target"].shape == (8, 2) and len(item["images"]) == 4
    assert "waypoints" in item["prompt"] and item["prompt"].count("(") >= 2
    w = ds.weights("equal")
    assert np.isclose(w[:len(ds.parts[0])].sum(), w[len(ds.parts[0]):].sum())
    assert fit_action_scale_frames(ds, cfg) >= 1.0


def test_lidar_tracks_find_a_walker_not_a_wall():
    from hnod import lidar_tracks as lt
    rng = np.random.default_rng(0)
    frames = []
    for i in range(120):  # 12 s at 10 Hz
        wall = np.c_[np.linspace(-5, 5, 200), np.full(200, 3.0)]
        person = np.array([-4 + 0.12 * i, 0.0]) + rng.normal(0, 0.1, (30, 2))  # 1.2 m/s along x
        xy = np.r_[wall, person]
        h = np.r_[rng.uniform(0.3, 1.9, 200), rng.uniform(0.3, 1.7, 30)]
        frames.append((xy, h, np.array([0.0, -2.0])))
    moving = lt.moving_points(frames)
    assert moving[60][200:].mean() > 0.8 and moving[60][:200].mean() < 0.05
    dets = [lt.clusters(xy[m], h[m]) for (xy, h, _), m in zip(frames, moving)]
    tracks = lt.track(dets, 0.1)
    assert len(tracks) == 1 and np.hypot(*(tracks[0]["xy"][-1] - tracks[0]["xy"][0])) > 8
