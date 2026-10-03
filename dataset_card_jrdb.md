---
license: cc-by-nc-sa-3.0
pretty_name: Robot Navigation Open Scenarios (JRDB)
task_categories:
- robotics
tags:
- navigation
- trajectory-prediction
- motion-forecasting
- social-navigation
- humanoid
- collision-avoidance
- lidar
- camera
size_categories:
- n<1K
configs:
- config_name: jrdb_2.5hz
  default: true
  data_files:
  - split: train
    path: data/jrdb_2.5hz/train-*.parquet
  - split: validation
    path: data/jrdb_2.5hz/validation-*.parquet
  - split: test
    path: data/jrdb_2.5hz/test-*.parquet
- config_name: jrdb_5hz
  data_files:
  - split: train
    path: data/jrdb_5hz/train-*.parquet
  - split: validation
    path: data/jrdb_5hz/validation-*.parquet
  - split: test
    path: data/jrdb_5hz/test-*.parquet
---

# Robot Navigation Open Scenarios: JRDB

Image-in, trajectory-out navigation scenarios with 3D ground truth, for evaluating a navigating
robot (the intended use is humanoid navigation). Converted from the training set of
[JRDB](https://jrdb.erc.monash.edu/) (JackRabbot Dataset, Martín-Martín et al.).

Each scenario has two halves:

- **Input, the past.** Forward-camera images of the 10 past frames and the current frame, plus the
  robot's own past positions. No obstacle information.
- **Ground truth, the future.** The robot's recorded trajectory over the 10 future frames, and for the
  current and each future frame the 3D geometry around it: boxes of every labelled pedestrian (static
  or dynamic) and a lidar point cloud with every point tagged ground / static / dynamic.

Same schema and evaluator as the CODa conversion,
[Jinyan0924/qwen_robotics_open_dataset](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset),
whose card documents every column, the coordinate frame, the point and map formats and the evaluation
protocol; this card lists what is different. Code: <https://github.com/naomili0924/qwen_robotics_open_dataset>.

![Example scenarios](figures/jrdb_2.5hz_examples.png)

*Three test scenarios (`jrdb_2.5hz`). Left: the current image with the recorded future path drawn on the
ground. Middle: bird's-eye view with the static map (dark = occupied), pedestrian boxes and their future
tracks, ego in green, goal as a star. Right: the lidar points of the last future step (black static,
red dynamic).*

## What is in it

| Config | Step | Past + future | Scenario spacing | train / validation / test |
|---|---|---|---|---|
| `jrdb_2.5hz` (default) | 0.4 s | 4 s + 4 s | 1 s | 272 / 38 / 118 |
| `jrdb_5hz` | 0.2 s | 2 s + 2 s | 1 s | 243 / 37 / 109 |

JRDB is labelled at 15 Hz. **Only scenarios in which the robot moves at least 1 m over the future
window are kept**: the robot stands still in 13 of the 27 sequences, and in a navigation benchmark
"stay where you are" is not a useful ground truth. Splits are by sequence: test =
`clark-center-2019-02-28_1`, `huang-basement-2019-01-25_0`, `packard-poster-session-2019-03-20_2`,
`tressider-2019-03-16_1`; validation = `gates-to-clark-2019-02-28_1`, `bytes-cafe-2019-02-07_0`; the
rest is train. JRDB's own test split has no public labels and is not used. A scenario is about
2.3 MB.

```python
from datasets import load_dataset
ds = load_dataset("Jinyan0924/qwen_robotics_open_dataset_jrdb", "jrdb_2.5hz", split="test")
```

```bash
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset_jrdb --config jrdb_2.5hz \
    --split test --pred my_predictions.json        # {"<scenario_id>": [[x, y] * 10], ...}
```

## Differences from the CODa conversion

- **Platform.** JackRabbot, a Segway-based robot about 0.65 x 0.55 m (nominal `ego` size, not
  published), driven by remote control indoors and on campus walkways. The ego reference point is the
  centre of the camera rig, 0.75 m above the floor.
- **Images.** `image_0` of the rig's lower camera ring, the one facing forward, 752 x 480, undistorted
  with the published calibration; `camera.K` describes the result.
- **Object types.** Pedestrians only (`PEDESTRIAN`), with JRDB's track ids. There are no static boxes:
  walls, furniture and poles exist only as lidar points and in the static map.
- **Scenario ids.** `jrdb_<sequence>_<frame>_<rate>hz`; `source_frames` is the JRDB frame number.
- **Occlusion** labels are not carried over; `occlusion` is 5 (unknown) wherever a box is valid.
- **Lidar.** Both 16-beam Velodynes merged. `static_map` merges sweeps from 5 s either side of the current
  step; an obstacle cell must be seen occupied at least 2 s apart. The sparse lidars leave much of the
  ground unobserved: 12% of the recorded path (2.5 Hz) lies in unknown cells.

## How it was built

1. **Source.** The 2019 training archive (`jrdb_train.zip`, 75 GB): 3D labels, timestamps, calibration,
   the forward camera and both lidars; the rosbags were not used.
2. **Odometry.** JRDB does not ship robot poses outside the rosbags. The wheel odometry pre-extracted
   for Google's Human Scene Transformer was used as a starting point and **refined by lidar scan
   matching** (point-to-plane ICP against a submap of the previous 10 sweeps, pedestrians cut out), because
   the wheel odometry drifts: seated people "moved" 0.43 m (median) / 1.6 m (90th pct) within 8 s under
   it, 0.17 m / 0.83 m after refinement. `scripts/refine_jrdb_odometry.py`.
3. **Lidar geometry.** The vertical lidar offsets in JRDB's `defaults.yaml` leave the two sweeps 0.95 m
   apart; the offsets were re-derived from the floor plane seen by each lidar (upper −1.24 m, lower
   −0.77 m in their own frames) and the floor level implied by the labels (box bottoms at −0.75 m).
4. **Scenarios.** Same pipeline as CODa: ground/obstacle split per sweep, merged maps, per-step labelled
   points, sliding windows in the robot frame of the current step.

## Baselines

Test split, collision rates in %, ADE/FDE in m, default evaluator settings (agent radius 0.3 m).
`any` = dynamic or static or map; `lidar` is the independent per-step point check.

| `jrdb_2.5hz` (4 s horizon) | any | dynamic | static | map | lidar | ADE | FDE |
|---|---|---|---|---|---|---|---|
| recorded path (`expert`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 |
| `straight_to_goal` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.18 | 0.00 |
| `constant_velocity` | 1.7 | 0.0 | 0.0 | 1.7 | 0.8 | 0.40 | 0.93 |
| `stationary` | 7.6 | 7.6 | 0.0 | 0.0 | 6.8 | 1.67 | 3.00 |

| `jrdb_5hz` (2 s horizon) | any | dynamic | static | map | lidar | ADE | FDE |
|---|---|---|---|---|---|---|---|
| recorded path (`expert`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 |
| `straight_to_goal` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.00 |
| `constant_velocity` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.12 | 0.29 |
| `stationary` | 1.8 | 1.8 | 0.0 | 0.0 | 0.9 | 0.93 | 1.68 |

The recorded path hits nothing (0.4% on train at 2.5 Hz), which is the label-noise floor here. With
only ~120 test scenarios, differences of one or two percentage points are within noise.

## Limitations

- **Small.** A few hundred scenarios per split; treat it as a complementary test set, not a training set.
- **Pedestrians only**, no other agent classes; static structure is lidar-derived.
- **Odometry is reconstructed.** Even after refinement, positions of static things drift by a few
  decimetres over a scenario; the merged static map is correspondingly blurred, especially outdoors
  (`clark-center`, `memorial-court`).
- **Lidar geometry is partly re-calibrated** from the data (see above); the yaw between the lidars is the
  published value.
- **One forward camera, not a humanoid recording, open loop, operator heuristic, derived geometry:** as
  for CODa.

## License and attribution

JRDB is published under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/); this
derived dataset carries the same license (non-commercial, share-alike). Changes from the original: labels
were re-sampled into fixed-length ego-centric windows, robot poses were reconstructed from odometry and
lidar, images were undistorted, lidar sweeps were thinned, labelled and merged into obstacle maps.
If you use this data, cite JRDB:

```bibtex
@article{martinmartin2021jrdb,
  title={JRDB: A Dataset and Benchmark of Egocentric Robot Visual Perception of Humans in Built Environments},
  author={Mart{\'\i}n-Mart{\'\i}n, Roberto and Patel, Mihir and Rezatofighi, Hamid and Shenoi, Abhijeet and Gwak, JunYoung and Frankel, Eric and Sadeghian, Amir and Savarese, Silvio},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```

The odometry starting point comes from the Human Scene Transformer release (Google Research).
