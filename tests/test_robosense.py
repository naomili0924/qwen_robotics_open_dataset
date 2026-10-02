"""Synthetic checks of RoboSense clip chaining and track re-association."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import robosense as rs  # noqa: E402


def frame(t, seq, objs, ego_x=0.0):
    """objs: list of (name, id, x, y) in the ego frame; ego sits at (ego_x, 0) heading +x."""
    n = len(objs)
    return dict(timestamp=int(t * 1e6), seq_token=seq, map_token="m", hs64_path=f"/d/hs64/{t}.bin",
                hs2global=np.eye(4), hs2livox=np.eye(4), ego2global_rotation=np.eye(3),
                images=dict(cams=dict(CAM_FRONT=dict(
                    data_path=f"/d/images/0/{t}.jpg", cam_intrinsic=np.eye(3).tolist(), cam_dist=[[0.0] * 5],
                    img_width=1920, img_height=1080, sensor2ego_rotation=np.eye(3), sensor2ego_translation=np.zeros(3)))), ego2global_translation=np.array([ego_x, 0.0, 0.0]),
                annos=dict(name=np.array([o[0] for o in objs]), id=np.array([o[1] for o in objs]),
                           location=np.array([[o[2], o[3], 0.0] for o in objs]).reshape(n, 3),
                           dimensions=np.tile([0.6, 0.5, 1.7], (n, 1)), rotation_y=np.zeros(n)))


def test_contiguous_clips_form_one_chain_and_gaps_split():
    frames = [frame(t, 1, []) for t in range(0, 5)] + [frame(t, 2, []) for t in range(5, 10)] \
        + [frame(t, 3, []) for t in range(30, 35)]
    chains = rs.build_chains(frames)
    assert [len(c) for c in chains] == [10, 5]


def test_tracks_are_reassociated_across_the_cut():
    # A pedestrian walks +y at 1 m/s and a parked car stays put; ids change at the clip boundary.
    a = [frame(t, 1, [("Pedestrian", 7, 5.0, float(t)), ("Car", 8, -6.0, 3.0)]) for t in range(0, 5)]
    b = [frame(t, 2, [("Car", 1, -6.0, 3.1), ("Pedestrian", 2, 5.0, float(t))]) for t in range(5, 10)]
    seg = rs.chain_to_segment(rs.build_chains(a + b)[0], "train", 0)
    assert len(seg["tracks"]) == 2
    for tr in seg["tracks"].values():
        assert tr["valid"].all()
    ped = next(tr for tr in seg["tracks"].values() if tr["category"] == "Pedestrian")
    assert np.allclose(ped["xyz"][:, 1], np.arange(10))
    assert np.allclose(ped["xyz"][:, 2], 0.85)                  # bottom-centre label -> box centre
    assert np.allclose(np.cos(ped["yaw"] - np.pi / 2), 1.0)     # heading follows the direction of travel


def test_far_objects_are_not_merged():
    a = [frame(t, 1, [("Pedestrian", 7, 5.0, 0.0)]) for t in range(0, 3)]
    b = [frame(t, 2, [("Pedestrian", 7, 15.0, 0.0)]) for t in range(3, 6)]  # same id, different person
    seg = rs.chain_to_segment(rs.build_chains(a + b)[0], "train", 0)
    assert len(seg["tracks"]) == 2


def test_label_id_is_ignored_when_the_implied_motion_is_impossible():
    # Inside one clip the same pedestrian id shows up 10 m away one second later: two people, two tracks.
    frames = [frame(0, 1, [("Pedestrian", 7, 5.0, 0.0)]), frame(1, 1, [("Pedestrian", 7, 5.0, 10.0)]),
              frame(2, 1, [("Pedestrian", 7, 5.0, 10.5)])]
    seg = rs.chain_to_segment(rs.build_chains(frames)[0], "train", 0)
    assert sorted(int(tr["valid"].sum()) for tr in seg["tracks"].values()) == [1, 2]


def test_duplicate_ids_in_a_frame_fall_back_to_position():
    # Two pedestrians share an id in every frame; each must still follow its own path.
    frames = [frame(t, 1, [("Pedestrian", 3, 5.0, 0.5 * t), ("Pedestrian", 3, -5.0, -0.5 * t)]) for t in range(6)]
    seg = rs.chain_to_segment(rs.build_chains(frames)[0], "train", 0)
    assert len(seg["tracks"]) == 2
    for tr in seg["tracks"].values():
        assert tr["valid"].all() and np.ptp(tr["xyz"][:, 0]) == 0
