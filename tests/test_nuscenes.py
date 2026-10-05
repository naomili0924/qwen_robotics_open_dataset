"""hnod.nuscenes on a synthetic dataset written in the nuScenes table format.

This checks the reader's geometry (frames, quaternion order, box size order) against values
built by hand; it cannot prove agreement with the real files.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hnod import nuscenes  # noqa: E402

F, SPEED, YAW = 26, 4.0, 0.5          # key frames, ego speed [m/s], ego heading in the global frame
LIDAR_Z = 1.84
WALL_Y = 6.0                          # a wall 6 m to the left of the lane, in the ego-aligned frame


def wxyz(yaw):
    x, y, z, w = Rotation.from_euler("z", yaw).as_quat()
    return [w, x, y, z]


def make_dataset(root):
    d = root / "v1.0-mini"
    d.mkdir(parents=True)
    (root / "samples/CAM_FRONT").mkdir(parents=True)
    (root / "samples/LIDAR_TOP").mkdir(parents=True)
    fwd, left = np.array([np.cos(YAW), np.sin(YAW)]), np.array([-np.sin(YAW), np.cos(YAW)])
    # optical frame (x right, y down, z forward) -> ego (x forward, y left, z up)
    cam_R = Rotation.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)).as_quat()
    tables = dict(
        sensor=[dict(token="s_cam", channel="CAM_FRONT", modality="camera"),
                dict(token="s_lid", channel="LIDAR_TOP", modality="lidar")],
        calibrated_sensor=[
            dict(token="c_cam", sensor_token="s_cam", translation=[1.7, 0.0, 1.5],
                 rotation=[cam_R[3], cam_R[0], cam_R[1], cam_R[2]], camera_intrinsic=[[1266, 0, 800], [0, 1266, 450], [0, 0, 1]]),
            dict(token="c_lid", sensor_token="s_lid", translation=[0.9, 0.0, LIDAR_Z], rotation=wxyz(0.0), camera_intrinsic=[])],
        category=[dict(token="cat_ped", name="human.pedestrian.adult"), dict(token="cat_car", name="vehicle.car")],
        instance=[dict(token="i_ped", category_token="cat_ped"), dict(token="i_car", category_token="cat_car")],
        visibility=[dict(token="4", level="v80-100")],
        scene=[dict(token="sc0", name="scene-0001", first_sample_token="smp0", last_sample_token=f"smp{F - 1}", nbr_samples=F)],
        sample=[], sample_data=[], ego_pose=[], sample_annotation=[])
    rng = np.random.default_rng(0)
    for i in range(F):
        t = i / 2.0
        ego_xy = 100.0 + fwd * SPEED * t
        tables["sample"].append(dict(token=f"smp{i}", timestamp=int(1.5e15 + t * 1e6), scene_token="sc0",
                                     prev=f"smp{i - 1}" if i else "", next=f"smp{i + 1}" if i < F - 1 else ""))
        tables["ego_pose"].append(dict(token=f"p{i}", translation=[ego_xy[0], ego_xy[1], 0.0], rotation=wxyz(YAW)))
        for ch, cal, ext in (("CAM_FRONT", "c_cam", "jpg"), ("LIDAR_TOP", "c_lid", "pcd.bin")):
            tables["sample_data"].append(dict(token=f"{ch}{i}", sample_token=f"smp{i}", ego_pose_token=f"p{i}",
                                              calibrated_sensor_token=cal, filename=f"samples/{ch}/{i:03d}.{ext}",
                                              is_key_frame=True, width=1600, height=900))
        cv2.imwrite(str(root / f"samples/CAM_FRONT/{i:03d}.jpg"), np.full((90, 160, 3), i, np.uint8))
        # lidar frame = ego frame shifted by (0.9, 0, LIDAR_Z): flat ground plus a wall at ego y = WALL_Y
        g = np.c_[rng.uniform(-25, 25, 6000), rng.uniform(-25, 25, 6000), np.full(6000, -LIDAR_Z)]
        w = np.c_[rng.uniform(-25, 25, 3000), np.full(3000, WALL_Y), rng.uniform(-LIDAR_Z + 0.3, 0.0, 3000)]
        pts = np.concatenate([g, w])
        np.c_[pts, np.zeros((len(pts), 2))].astype(np.float32).tofile(root / f"samples/LIDAR_TOP/{i:03d}.pcd.bin")
        # a pedestrian walking across, 3 m to the right at 20 m ahead of the start; a parked car
        ped = 100.0 + fwd * 20.0 + left * (-3.0 + 1.0 * t)
        car = 100.0 + fwd * 30.0 + left * 3.0
        tables["sample_annotation"] += [
            dict(token=f"a_ped{i}", sample_token=f"smp{i}", instance_token="i_ped", visibility_token="4",
                 translation=[ped[0], ped[1], 0.9], size=[0.6, 0.7, 1.8], rotation=wxyz(YAW + np.pi / 2)),
            dict(token=f"a_car{i}", sample_token=f"smp{i}", instance_token="i_car", visibility_token="4",
                 translation=[car[0], car[1], 0.8], size=[1.9, 4.5, 1.6], rotation=wxyz(YAW))]
    for name, rows in tables.items():
        json.dump(rows, open(d / f"{name}.json", "w"))


def test_reader_and_conversion(tmp_path):
    import convert_nuscenes as conv
    from hnod import eval as ev
    make_dataset(tmp_path)
    tables = nuscenes.Tables(tmp_path, "v1.0-mini")
    seg, = nuscenes.load_segments(tables, "scene-0001")
    assert len(seg["frames"]) == F and seg["camera"]["width"] == 1600
    car = seg["tracks"]["i_car"]
    assert np.allclose(car["lwh"][0], (4.5, 1.9, 1.6))                       # (width, length, height) -> l, w, h
    assert np.isclose(car["yaw"][0], YAW)
    assert nuscenes.type_of("human.pedestrian.adult") == "PEDESTRIAN" and nuscenes.type_of("vehicle.car") == "VEHICLE"
    # camera looks along ego +x
    ego_from_cam = np.linalg.inv(seg["camera"]["T_camera_from_ego"])
    assert np.allclose(ego_from_cam[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)

    conv._tables = tables
    scene, split, n = conv.convert_scene(("scene-0001", str(tmp_path / "out"), "train"))
    assert n == 3                                                             # anchors 10, 12, 14 of 26 frames
    from datasets import load_dataset
    files = sorted(str(f) for f in (tmp_path / "out" / conv.CONFIG / "_parts/train").glob("*.parquet"))
    row = load_dataset("parquet", data_files=files, split="train")[0]
    ego = row["ego"]
    assert np.allclose(np.diff(ego["x"]), SPEED / 2, atol=1e-3) and np.allclose(ego["y"], 0, atol=1e-3)
    assert row["task"] == "pointgoal" and row["instruction"] == ""
    tr = row["future_tracks"]
    types = list(tr["object_type"])
    ped, car = types.index("PEDESTRIAN"), types.index("VEHICLE")
    # at the current step (t = 5 s) the ego has driven 20 m: the pedestrian is abeam, 2 m to the left
    assert np.allclose([tr["x"][ped][0], tr["y"][ped][0]], [0.0, 2.0], atol=0.05)
    assert np.allclose([tr["x"][car][0], tr["y"][car][0]], [10.0, 3.0], atol=0.05)
    assert tr["is_stationary"][car] and not tr["is_stationary"][ped]
    # the wall: static points at y = WALL_Y, ground points at z = 0
    lid = row["future_lidar"]
    y, z, lab = (np.array(lid[k][0]) for k in ("y", "z", "label"))
    assert np.abs(np.median(y[lab == 1]) / 100 - WALL_Y) < 0.1
    assert np.abs(np.median(z[lab == 0]) / 100) < 0.05
    # driving straight on hits nothing; swerving left into the wall does
    r = ev.evaluate_scenario(row, ev.baseline_expert(row))
    assert not r["collided"] and not r["collided_lidar"]
    left = np.stack([np.linspace(1, 10, 10), np.linspace(0.7, 7.0, 10)], 1)
    assert ev.evaluate_scenario(row, left)["collided_map"]
