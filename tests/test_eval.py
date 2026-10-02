"""Synthetic checks of the collision evaluator (run: python -m pytest tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import eval as ev  # noqa: E402
from hnod.io import png_bytes  # noqa: E402
from hnod.maps import MAP_SIZE, PIX_FREE, PIX_OCCUPIED, xy_to_rowcol  # noqa: E402
from hnod.scenario import CURRENT, N_FUTURE, N_STEPS  # noqa: E402

NAN = float("nan")


def make_row(tracks=(), occupied_xy=(), rate_hz=2.0):
    """Ego drives straight ahead at 1 m/s; tracks are dicts with x, y (N_STEPS,), type, stationary."""
    t = (np.arange(N_STEPS) - CURRENT) / rate_hz
    img = np.full((MAP_SIZE, MAP_SIZE), PIX_FREE, np.uint8)
    for x, y in occupied_xy:
        r, c = xy_to_rowcol(x, y)
        img[int(round(r)), int(round(c))] = PIX_OCCUPIED
    cols = {k: [] for k in ("id", "category", "object_type", "length", "width", "height", "is_stationary",
                            "is_operator", "x", "y", "z", "heading", "vx", "vy", "valid", "occlusion")}
    for i, tr in enumerate(tracks):
        valid = np.isfinite(tr["x"])
        cols["id"].append(str(i)); cols["category"].append(tr["type"]); cols["object_type"].append(tr["type"])
        cols["length"].append(tr.get("length", 0.6)); cols["width"].append(tr.get("width", 0.6)); cols["height"].append(1.7)
        cols["is_stationary"].append(tr.get("stationary", False)); cols["is_operator"].append(tr.get("operator", False))
        cols["x"].append(list(tr["x"])); cols["y"].append(list(tr["y"])); cols["z"].append([0.85] * N_STEPS)
        cols["heading"].append([0.0] * N_STEPS); cols["vx"].append([0.0] * N_STEPS); cols["vy"].append([0.0] * N_STEPS)
        cols["valid"].append(valid.tolist()); cols["occlusion"].append([0] * N_STEPS)
    return dict(scenario_id="synthetic", rate_hz=rate_hz, tracks=cols, static_map={"bytes": png_bytes(img)},
                ego=dict(x=t.tolist(), y=[0.0] * N_STEPS, vx=[1.0] * N_STEPS, vy=[0.0] * N_STEPS), goal=[t[-1], 0.0])


def straight(row):
    return ev.baseline_expert(row)


def test_free_space_has_no_collision():
    row = make_row()
    res = ev.evaluate_scenario(row, straight(row))
    assert not res["collided"] and res["ade"] == 0 and res["fde"] == 0


def test_box_distance():
    assert ev.box_distance(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0) == 1.0          # 1 m beyond the front face
    assert ev.box_distance(0.2, 0.1, 0.0, 0.0, 0.0, 2.0, 1.0) == 0.0          # inside
    assert np.isclose(ev.box_distance(0.0, 2.0, 0.0, 0.0, np.pi / 2, 2.0, 1.0), 1.0)  # rotated: long side along y


def test_crossing_pedestrian_is_a_dynamic_collision():
    # Pedestrian walks across the ego path and is on it exactly when the ego gets there (x = 2.5 m at t = 2.5 s).
    t = (np.arange(N_STEPS) - CURRENT) / 2.0
    row = make_row([dict(type="PEDESTRIAN", x=np.full(N_STEPS, 2.5), y=(t - 2.5) * 1.0)])
    res = ev.evaluate_scenario(row, straight(row))
    assert res["collided_dynamic"] and not res["collided_static"] and not res["collided_map"]
    assert np.isclose(res["first_collision_s"], 2.2, atol=0.31)
    # The same pedestrian crossing 3 s later is missed.
    row = make_row([dict(type="PEDESTRIAN", x=np.full(N_STEPS, 2.5), y=(t - 5.5) * 1.0)])
    assert not ev.evaluate_scenario(row, straight(row))["collided"]


def test_collision_between_steps_is_caught():
    # At 2 Hz the pedestrian is 0.9 m to either side of the path at consecutive steps but crosses it in between.
    t = (np.arange(N_STEPS) - CURRENT) / 2.0
    y = np.where(t <= 2.0, -0.9, 0.9) + 0.0
    row = make_row([dict(type="PEDESTRIAN", x=np.full(N_STEPS, 2.25), y=y)])
    assert ev.evaluate_scenario(row, straight(row))["collided_dynamic"]


def test_stationary_track_blocks_even_when_no_longer_visible():
    x = np.full(N_STEPS, 3.0)
    x[CURRENT + 1:] = NAN  # standing person seen only up to the current step
    row = make_row([dict(type="PEDESTRIAN", x=x, y=np.zeros(N_STEPS), stationary=True)])
    res = ev.evaluate_scenario(row, straight(row))
    assert res["collided_static"] and not res["collided_dynamic"]


def test_operator_and_start_overlap_are_ignored():
    behind = dict(type="PEDESTRIAN", x=(np.arange(N_STEPS) - CURRENT) / 2.0 - 0.4, y=np.zeros(N_STEPS))
    row = make_row([dict(behind, operator=True)])
    assert not ev.evaluate_scenario(row, straight(row))["collided"]
    row = make_row([behind])  # not flagged, but already overlapping at the current step
    res = ev.evaluate_scenario(row, straight(row))
    assert res["start_overlap"] and not res["collided"]


def test_map_obstacle_and_static_boxes():
    row = make_row(occupied_xy=[(3.0, 0.1)])
    res = ev.evaluate_scenario(row, straight(row))
    assert res["collided_map"] and not res["collided_dynamic"]
    assert not ev.evaluate_scenario(row, straight(row), use_map=False)["collided"]
    # A STATIC-type box is left to the map when the map is used, and checked as a box otherwise.
    row = make_row([dict(type="STATIC", x=np.full(N_STEPS, 3.0), y=np.zeros(N_STEPS), stationary=True)])
    assert not ev.evaluate_scenario(row, straight(row))["collided"]
    assert ev.evaluate_scenario(row, straight(row), use_map=False)["collided_static"]


def test_baseline_shapes():
    row = make_row()
    for fn in ev.BASELINES.values():
        assert fn(row).shape == (N_FUTURE, 2)
    assert np.allclose(ev.baseline_constant_velocity(row), ev.baseline_expert(row))
