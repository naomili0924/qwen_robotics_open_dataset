import numpy as np
from datasets import Dataset

from hnod import frames as fr
from hnod.windows import FrameWindows, WindowConfig


def _episode(eid, embodiment, speed, n=80, rate=5.0, stop_after=None):
    t = np.arange(n) / rate
    d = speed * np.minimum(t, stop_after if stop_after is not None else t[-1])
    rows = []
    for i in range(n):
        img = {"bytes": fr.jpeg_bytes(np.full((8, 8, 3), i, np.uint8)), "path": None}  # brightness = frame number
        rows.append(dict(episode_id=eid, frame_index=i, source_frame=i, timestamp=float(t[i]), image=img,
                         image_right=None, pose=[float(d[i]), 0.0, 0.0, 0.0]))
    ep = dict(episode_id=eid, dataset="synthetic", source_sequence=eid, split="train", num_frames=n, rate_hz=rate,
              duration_s=float(t[-1]), path_length_m=float(d[-1]), median_speed_mps=speed, embodiment=embodiment,
              environment="indoor", environment_method="labelled", indoor_fraction=1.0, frame_indoor_prob=[1.0] * n,
              camera=dict(name="front", width=8, height=8, K=[1, 0, 4, 0, 1, 4, 0, 0, 1], distortion_model="none",
                          distortion=[], height_m=1.3, stereo_baseline_m=float("nan")),
              segments=[], ends_at_rest=False, fits="both", licence="test")
    return rows, ep


def _windows(**kw):
    r1, e1 = _episode("human", "person_walking", 1.2, **kw)
    r2, e2 = _episode("robot", "wheeled_robot", 0.5)
    return FrameWindows(Dataset.from_list(r1 + r2, features=fr.FRAME_FEATURES),
                        Dataset.from_list([e1, e2], features=fr.EPISODE_FEATURES),
                        WindowConfig(mode="final_frame", horizon_s=5.0, n_waypoints=10, n_past=5, past_dt_s=1.0))


def test_targets_are_spaced_in_time_so_speed_is_part_of_the_answer():
    w = _windows()
    human = next(w.sample(k) for k in range(len(w)) if w.sample(k)["episode_id"] == "human")
    robot = next(w.sample(k) for k in range(len(w)) if w.sample(k)["episode_id"] == "robot")
    assert np.allclose(human["target"][:, 0], 1.2 * 0.5 * np.arange(1, 11), atol=1e-5)
    assert np.allclose(robot["target"][:, 0], 0.5 * 0.5 * np.arange(1, 11), atol=1e-5)
    assert human["prompt"] == ("The camera is carried by a human walking. The past views are sampled at 1 Hz. "
                               "Please predict the next 10 positions at 2 Hz.")
    assert robot["prompt"].startswith("The camera is carried by a robot.")


def test_only_frames_with_a_full_horizon_are_samples_and_the_final_frame_is_5_s_ahead():
    w = _windows()
    assert len(w) == 2 * (80 - 25)  # 5 s at 5 Hz = 25 frames must follow
    item = w[30]
    s = w.sample(30)
    assert s["final_row"] - s["row"] == 25 and abs(s["final_dt_s"] - 5.0) < 1e-6
    assert len(item["images"]) == 6  # 5 past + now
    assert np.asarray(item["final_image"])[0, 0, 0] == np.asarray(item["images"][-1])[0, 0, 0] + 25


def test_a_stop_is_kept_as_motion():
    w = _windows(stop_after=4.0)  # the human stops 4 s into the recording
    s = next(w.sample(k) for k in range(len(w)) if w.sample(k)["episode_id"] == "human" and w.sample(k)["frame_index"] == 10)
    x = s["target"][:, 0]  # at 2 s: walks 2 more seconds (2.4 m), then stands
    assert np.isclose(x[3], 2.4, atol=1e-4) and np.allclose(x[3:], 2.4, atol=1e-4)


def test_item_has_no_numbers_only_images_and_embodiment():
    from vla.config import Config
    from vla.data import FrameNavDataset, window_config
    cfg = Config(data_format="frames", window_mode="final_frame", frames=7, horizon=10, action_scale=2.0,
                 kinematic_input=False, ego_history=False)
    wc = window_config(cfg)
    assert wc.mode == "final_frame" and wc.n_past == 5 and wc.n_waypoints == 10
    r, e = _episode("robot", "wheeled_robot", 0.5)
    w = FrameWindows(Dataset.from_list(r, features=fr.FRAME_FEATURES), Dataset.from_list([e], features=fr.EPISODE_FEATURES), wc)
    item = FrameNavDataset([w], cfg)[0]
    assert len(item["images"]) == 7 and len(item["image_tags"]) == 7 and item["image_tags"][-1] == "View in 5 s:"
    assert item["prompt"].startswith("The camera is carried by a robot.") and not item["kin"].any()
    assert item["target"].shape == (10, 2) and "(" not in item["prompt"]


def test_timed_scoring_completion_and_resampling():
    from hnod import suite
    from hnod.scenario import N_FUTURE
    n = N_FUTURE + 1
    ego = dict(x=[0.5 * (i - 10) for i in range(21)], y=[0.0] * 21, z=[0.0] * 21, heading=[0.0] * 21,
               vx=[1.0] * 21, vy=[0.0] * 21, length=0.6, width=0.6, height=1.7)   # 1 m/s at 2 Hz
    tr = dict(id=[], category=[], object_type=[], length=[], width=[], height=[], is_stationary=[], is_operator=[],
              x=[], y=[], z=[], heading=[], vx=[], vy=[], valid=[], occlusion=[])
    row = dict(scenario_id="t", rate_hz=2.0, ego=ego, future_tracks=tr, goal=[5.0, 0.0], static_map=None,
               future_lidar=None, final_step=10, final_xy=[5.0, 0.0])
    b = suite.timed_baselines(row)
    assert suite.score_timed(row, b["recorded"])["completed"] and suite.score_timed(row, b["recorded"])["ade"] < 1e-6
    assert suite.score_timed(row, b["constant_velocity"])["success"]
    assert not suite.score_timed(row, b["stationary"])["completed"]
    row5 = dict(row, final_step=5, final_xy=[2.5, 0.0])     # a 10 s scenario scored on its first 5 steps
    steps = suite.steps_from_timed(suite.timed_baselines(row5)["recorded"], 5)
    assert np.allclose(steps[:5, 0], [0.5, 1.0, 1.5, 2.0, 2.5]) and np.allclose(steps[5:, 0], 2.5)


def test_camera_sentence_uses_stored_calibration_only_and_says_unknown_otherwise():
    from hnod.windows import camera_prompt, embodiment_prompt
    cam = dict(width=960, height=600, K=[368.0, 0, 480, 0, 368.0, 300, 0, 0, 1], height_m=1.25, height_source="dataset")
    t = camera_prompt(cam)
    assert "horizontal field of view is 105 degrees" in t and "mounted 1.25 m above the ground" in t
    # an estimated height (or no source) is never passed on as a fact
    assert "height above the ground is unknown" in camera_prompt(dict(cam, height_source="estimated"))
    assert "height above the ground is unknown" in camera_prompt(dict(cam, height_m=float("nan"), height_source="dataset"))
    assert camera_prompt({}) == "The camera's field of view is unknown; its height above the ground is unknown."
    assert camera_prompt(cam) in embodiment_prompt("person_walking", camera=cam)
    assert "field of view" not in embodiment_prompt("person_walking")
