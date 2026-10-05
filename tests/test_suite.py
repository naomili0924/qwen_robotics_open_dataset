import numpy as np

from hnod import suite
from hnod.scenario import N_FUTURE


def _row(ped_x, ped_y, rate=2.0, goal=(10.0, 0.0)):
    """One pedestrian with the given future positions (current step + N_FUTURE), no map, no lidar."""
    n = N_FUTURE + 1
    tr = dict(id=["p"], category=["Pedestrian"], object_type=["PEDESTRIAN"], length=[0.6], width=[0.6], height=[1.7],
              is_stationary=[False], is_operator=[False], x=[list(ped_x)], y=[list(ped_y)], z=[[0.85] * n],
              heading=[[0.0] * n], vx=[[0.0] * n], vy=[[0.0] * n], valid=[[True] * n], occlusion=[[0] * n])
    ego = dict(x=[0.0] * 21, y=[0.0] * 21, z=[0.0] * 21, heading=[0.0] * 21, vx=[0.0] * 21, vy=[0.0] * 21,
               length=0.6, width=0.6, height=1.7)
    return dict(scenario_id="t", rate_hz=rate, ego=ego, future_tracks=tr, goal=list(goal), static_map=None,
                future_lidar=None)


def test_timed_positions_follow_the_path_at_robot_speed():
    p = suite.timed_positions([[1.0, 0.0], [1.0, 1.0]], rate_hz=2.0, speed=0.5, n=10)
    assert np.allclose(p[0], [0.25, 0.0]) and np.allclose(p[3], [1.0, 0.0])
    assert np.allclose(p[-1], [1.0, 1.0])  # the path ends after 2 m: the robot stops there


def test_walked_into_from_behind_is_not_at_fault():
    t = np.arange(N_FUTURE + 1) / 2.0
    row = _row(-3.0 + 1.5 * t, np.zeros_like(t))  # a faster walker catching up from behind
    straight = np.c_[np.linspace(0.25, 10, 40), np.zeros(40)]
    res = suite.score(row, straight)
    assert res["collided_dynamic_any"] and not res["collided_dynamic"] and not res["collided"]


def test_walking_into_someone_ahead_is_at_fault():
    t = np.arange(N_FUTURE + 1) / 2.0
    row = _row(4.0 - 0.5 * t, np.zeros_like(t))  # oncoming
    res = suite.score(row, np.c_[np.linspace(0.25, 10, 40), np.zeros(40)])
    assert res["collided_dynamic"] and res["collided"]


def test_a_standing_robot_is_never_at_fault():
    t = np.arange(N_FUTURE + 1) / 2.0
    row = _row(4.0 - 1.0 * t, np.zeros_like(t))  # walks straight through the robot's position
    res = suite.score(row, np.zeros((1, 2)))
    assert not res["collided_dynamic"]


def test_wiggle_ignores_turns_but_not_zigzags():
    turn = np.r_[np.c_[np.linspace(0.1, 2, 20), np.zeros(20)], np.c_[np.full(20, 2.0), np.linspace(0.1, 2, 20)]]
    zig = np.c_[np.linspace(0.1, 4, 40), 0.15 * (np.arange(40) % 2)]
    assert suite.wiggle(turn) < 0.3 and suite.wiggle(zig) > 5


def test_progress_ratio():
    row = _row(np.full(N_FUTURE + 1, 50.0), np.zeros(N_FUTURE + 1))
    res = suite.score(row, np.c_[np.linspace(0.25, 10, 40), np.zeros(40)])
    assert np.isclose(res["progress_ratio"], 1.0) and res["success"]
    res = suite.score(row, np.zeros((1, 2)))
    assert res["progress_ratio"] == 0 and not res["success"]
