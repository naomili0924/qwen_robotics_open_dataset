---
license: cc0-1.0
task_categories:
- robotics
tags:
- navigation
- social-navigation
- indoor
- evaluation
- lidar
- camera
pretty_name: Robot Navigation Open Scenarios (MuSoHu, indoor walks)
size_categories:
- n<1K
configs:
- config_name: musohu_2hz
  default: true
  data_files:
  - split: test
    path: data/musohu_2hz/test-*.parquet
---

# Robot Navigation Open Scenarios: MuSoHu (indoor walks)

Image-in, trajectory-out navigation scenarios with 3D ground truth, converted from walks of the
[Multi-Modal Social Human Navigation dataset (MuSoHu)](https://doi.org/10.13021/orc2020/HZI4LJ) (CC0) through
George Mason University buildings (Johnson Center food court, Engineering, Horizon Hall, the Hub), a shopping mall
and a grocery store. It supplies the **real indoor scenes with people** of the evaluation suite
[qwen_robotics_nav_eval](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_nav_eval); the whole source is
reserved for evaluation (single `test` split). Same schema as the
[CODa scenarios](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset); converter
`scripts/convert_musohu.py`.

## How it differs from the robot datasets

- **A walking person, not a robot.** A helmet carries a Velodyne VLP-16 lidar and a ZED 2 camera at head height
  (the lidar is about 1.65 m above the floor), so the images are close to a humanoid's view. The walker moves at
  about 1.2 m/s.
- **People are tracked in the lidar, not annotated** (`hnod/lidar_tracks.py`). Obstacle points 0.3-2 m above the
  floor whose 0.2 m cell is occupied for less than 2 s within +-5 s are *moving*; they are clustered into
  person-sized blobs and linked over time; tracks seen for at least 0.8 s that travel at least 0.6 m are
  `PEDESTRIAN`. People who stand still for seconds become part of the static map instead. Expect missed people
  (sparse 16-beam returns beyond about 10 m, occlusion) and occasional false tracks (doors, carts).
- **Levelling from the floor.** The helmet pitches and rolls; each sweep is levelled with the floor plane fitted in
  that sweep (RANSAC, smoothed over +-0.5 s), and the walker's position and heading come from the ZED odometry.
  The scenario frame is gravity-aligned; `camera.T_scenario_from_camera` is given per image because the camera
  tilts with the head.
- **Blind zone.** The lowest lidar beam meets the floor about 6 m away, so low objects next to the walker are seen
  only from farther away earlier; the static map merges +-5 s of sweeps to cover them.
- **Odometry, not ground truth.** ZED visual-inertial odometry drifts slowly; maps merge only +-5 s.

## Contents

2 Hz steps (5 s past + current + 5 s future), one scenario per second where the walker moves at least 1 m.
Recordings were chosen by place name (indoor buildings); the suite labels each scenario indoor or outdoor by eye.

## Licence

MuSoHu is CC0; this conversion is released under CC0 too. Cite the MuSoHu authors (Nguyen et al., "Toward Human-Like
Social Robot Navigation: A Large-Scale, Multi-Modal, Social Human Navigation Dataset", IROS 2023).
