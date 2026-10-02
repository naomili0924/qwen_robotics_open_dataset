---
license: cc-by-nc-sa-4.0
pretty_name: Robot Navigation Open Scenarios (CODa)
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
size_categories:
- 1K<n<10K
configs:
- config_name: coda_2hz
  default: true
  data_files:
  - split: train
    path: data/coda_2hz/train-*.parquet
  - split: validation
    path: data/coda_2hz/validation-*.parquet
  - split: test
    path: data/coda_2hz/test-*.parquet
- config_name: coda_10hz
  data_files:
  - split: train
    path: data/coda_10hz/train-*.parquet
  - split: validation
    path: data/coda_10hz/validation-*.parquet
  - split: test
    path: data/coda_10hz/test-*.parquet
---

# Robot Navigation Open Scenarios

Waymo-Open-Motion-style scenarios for evaluating a navigating robot (the intended use is
humanoid navigation): each scenario gives **10 past steps, the current step and 10 future
steps** of the ego robot and of every tracked object around it, plus a static obstacle map.
A planner predicts the ego's 10 future positions and is scored on whether that path
collides with future positions of moving agents, with stationary objects, or with static
structure.

This release converts the [UT Campus Object Dataset (CODa)](https://amrl.cs.utexas.edu/coda/).
Conversion and evaluation code: <https://github.com/naomili0924/qwen_robotics_open_dataset>.

![Example scenarios](figures/coda_2hz_examples.png)

*Six test scenarios (`coda_2hz`). Green: ego past (faint) and future (dotted), star = goal.
Red: pedestrians, orange: bikes/scooters, purple: vehicles, blue: static objects, grey box:
robot operator. Background: static map (dark = occupied, white = free, grey = unknown).*

## Configurations

| Config | Step | History + future | Scenario spacing | train / validation / test |
|---|---|---|---|---|
| `coda_2hz` (default) | 0.5 s | 5 s + 5 s | 1 s | 1,553 / 86 / 322 |
| `coda_10hz` | 0.1 s | 1 s + 1 s | 0.5 s | 3,917 / 300 / 1,027 |

Both have 21 steps per scenario with `current_time_index = 10`. `coda_10hz` is the direct
analogue of Waymo's 10 Hz sampling, but its 1 s horizon is too short to separate planners
on collisions (see baselines below). **Use `coda_2hz` for navigation evaluation.**

Splits are by whole CODa sequence, so no recording is shared between splits:
validation = sequences 4, 10, 19; test = 1, 7, 9, 12, 17; train = the rest.
The same campus routes appear in several sequences, so places (not recordings) do repeat across splits.

## Quick start

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("Jinyan0924/qwen_robotics_open_dataset", "coda_2hz", split="test")
row = ds[0]
ego_xy = np.stack([row["ego"]["x"], row["ego"]["y"]], axis=1)      # (21, 2)
history, future = ego_xy[:11], ego_xy[11:]                          # steps 0..10 / 11..20
agents_x = np.array(row["tracks"]["x"], dtype=float)                # (num_tracks, 21), NaN where not valid
static_map = np.array(row["static_map"])                            # (400, 400) uint8
```

Scoring predictions (with the code repository checked out):

```bash
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz \
    --split test --pred my_predictions.json        # {"<scenario_id>": [[x, y] * 10], ...}
```

## Coordinate frame

Everything in a scenario is expressed in one ego-centric frame fixed at the current step:
origin on the ground under the robot at the current step, **+x = robot heading, +y = left,
+z = up**, metres and radians. `heading` is counter-clockwise from +x. x and y are horizontal
offsets in the source world frame; z is height above the plane the robot stands on.
`world_from_scenario` (row-major 4x4) maps scenario x, y back to CODa's world frame.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `scenario_id` | string | `coda_<sequence>_<frame of current step>_<rate>hz` |
| `dataset`, `sequence`, `segment` | string, string, int | source dataset, CODa sequence, index of the contiguously labelled segment |
| `rate_hz` | float | steps per second (2 or 10) |
| `current_time_index` | int | always 10 |
| `timestamps` | float64[21] | Unix time of each step |
| `source_frames` | int[21] | CODa frame index of each step |
| `world_from_scenario` | float64[16] | scenario frame to CODa world frame |
| `ego` | struct | `x, y, z, heading, vx, vy` (each float[21]) and `length, width, height` of the recording robot (0.99 x 0.67 x 1.0 m). `z` is the box centre. The future part is the recorded (human-teleoperated) path. |
| `goal` | float[2] | ego (x, y) at the last future step |
| `tracks` | struct of lists | one entry per tracked object, see below |
| `tracks_to_predict` | int[] | indices of movable, non-operator tracks that are valid at the current step and at some future step |
| `static_map` | image | static obstacle map built from lidar within +-5 s of the current step. **Evaluation ground truth; contains future observations, do not feed it to a model.** |
| `observed_map` | image | same, built only from lidar up to the current step (past 5 s). Valid model input. |
| `map_resolution` | float | 0.1 m per pixel |
| `num_tracks`, `num_pedestrians` | int | tracks in the scenario; non-operator pedestrians valid at the current step |
| `ego_speed`, `ego_future_distance` | float | ego speed at the current step (m/s); recorded path length over the future (m) |

### `tracks`

| Field | Shape | Meaning |
|---|---|---|
| `id` | [N] | source instance id; a `#k` suffix marks a track that was split because the source reused the id for another object |
| `category` | [N] | CODa class name (52 classes, e.g. `Pedestrian`, `Bike`, `Pole`, `Tree`) |
| `object_type` | [N] | `PEDESTRIAN`, `CYCLE` (bike, scooter, motorcycle, skateboard, segway), `VEHICLE`, `OTHER_MOVABLE`, or `STATIC` |
| `length`, `width`, `height` | [N] | box size (median over the segment) |
| `is_stationary` | [N] | `STATIC` type, or stays within 0.25 m of its median position during the scenario |
| `is_operator` | [N] | the robot's human operator, who walks next to it (see Limitations) |
| `x`, `y`, `z`, `heading` | [N, 21] | box centre and yaw per step, NaN where not valid |
| `vx`, `vy` | [N, 21] | velocity by central differences over up to +-0.3 s |
| `valid` | [N, 21] | whether the object is labelled at that step |
| `occlusion` | [N, 21] | CODa occlusion label: 0 none, 1 light, 2 medium, 3 heavy, 4 full, 5 unknown, -1 not valid |

### Map images

400 x 400 pixels, 0.1 m per pixel, covering 20 m around the ego. Pixel values: **0 = occupied,
255 = observed free ground, 127 = unknown**. The ego is at the image centre, +x is up and +y is left:

```python
x = 20.0 - (row_index + 0.5) * 0.1
y = 20.0 - (col_index + 0.5) * 0.1
```

Occupied means lidar returns between 0.2 m and 2.0 m above the local ground that persist for at
least 2 s. Movable objects (people, bikes, vehicles) are cut out of the map and represented only by
their boxes.

## Evaluation protocol

`hnod/eval.py` in the code repository scores a predicted path of 10 future (x, y) positions:

- The agent is a disc of radius 0.3 m (configurable). Paths are checked every 0.1 s, interpolating
  between steps.
- **dynamic**: moving tracks at their recorded future poses. Pedestrians are discs of radius 0.3 m
  (their labelled boxes are loose, about 1 m across); other objects use their boxes.
- **static**: stationary movable-type tracks (standing people, parked bikes and cars), assumed to stay
  where they were last seen.
- **map**: occupied cells of `static_map`. Boxes of `STATIC`-type tracks are not used when the map is,
  because the map already holds those objects with their true ground-level footprint.
- Ignored: the operator, obstacles already overlapping the ego at the current step, and boxes whose
  underside is more than 2 m above the ground.
- Reported: collision rate (any / dynamic / static / map), ADE and FDE against the recorded path,
  time of first collision, and the fraction of the path in unknown space.

Baselines on the test split (collision rates in %, ADE/FDE in m):

| `coda_2hz` (5 s horizon) | any | dynamic | static | map | ADE | FDE |
|---|---|---|---|---|---|---|
| recorded path (`expert`) | 1.2 | 0.3 | 0.3 | 0.6 | 0.00 | 0.00 |
| `constant_velocity` | 4.7 | 1.9 | 0.3 | 2.5 | 0.22 | 0.47 |
| `straight_to_goal` | 4.0 | 0.3 | 0.3 | 3.7 | 0.07 | 0.00 |
| `stationary` | 8.4 | 8.4 | 0.0 | 0.0 | 2.52 | 4.59 |

| `coda_10hz` (1 s horizon) | any | dynamic | static | map | ADE | FDE |
|---|---|---|---|---|---|---|
| recorded path (`expert`) | 0.3 | 0.2 | 0.0 | 0.1 | 0.00 | 0.00 |
| `constant_velocity` | 0.3 | 0.2 | 0.1 | 0.0 | 0.03 | 0.06 |
| `straight_to_goal` | 0.3 | 0.2 | 0.0 | 0.1 | 0.01 | 0.00 |
| `stationary` | 1.6 | 1.6 | 0.0 | 0.0 | 0.49 | 0.90 |

The recorded path did not actually hit anything, so its collision rate is the label-noise floor
of the benchmark: 1.2% on `coda_2hz` test (2.1% on train, 0% on validation) and 0.3% on `coda_10hz`
test. Differences smaller than that are not meaningful.

## How it was built

1. **Labels.** CODa's 3D boxes (27,863 frames at 10 Hz in 21 sequences), lidar poses and timestamps.
   Only these and the lidar sweeps of labelled frames are used; no images.
2. **Segments.** CODa is labelled in bursts of consecutive frames; instance ids are only consistent
   inside a burst. Each burst is one segment (83 in total). Twelve single dropped label frames inside
   segments are filled by interpolating their neighbours.
3. **Track clean-up.** Boxes with non-finite or absurd coordinates are dropped. A track is split
   where consecutive observations are further apart than its type can move, because ids are
   sometimes reused for a different object.
4. **Static map.** Each lidar sweep is split into ground and obstacle points by growing a ground
   surface outwards from the robot, then rasterised. Sweeps are merged per scenario after removing
   labelled movable objects. Checked against CODa's terrain labels (4,980 frames): on paved and
   tiled surfaces 99.1% of points are classified as ground and 0.16% as obstacle. Grass is the
   exception: 45% of grass points are classified as obstacle (see Limitations).
5. **Scenarios.** Sliding windows over each segment, transformed to the ego frame at the current step.

## Limitations

- **Not recorded by a humanoid.** The platform is a wheeled Clearpath Husky with the lidar 0.8 m above
  the ground, teleoperated at about 0.9 m/s. Viewpoint, speed and reachable terrain differ from a humanoid.
- **Open loop.** Other agents replay their recorded motion and do not react to the predicted path.
  A path that deviates far from the recorded one is judged against people who were reacting to a
  robot that was somewhere else.
- **Operator.** One or two people walk right next to the robot throughout CODa. They are detected
  by a heuristic (a pedestrian that stays within 2.5 m of the robot) and flagged `is_operator`; the
  evaluator ignores them. The heuristic can miss an operator or flag a pedestrian who walks alongside.
- **Static map is derived, not labelled.** Lawns are often marked occupied (45% of grass points):
  on this campus they are mostly raised or banked beside the paths, and tall grass also counts. Stairs
  and slopes steeper than roughly 40% are partly marked occupied; drop-offs are not detected; objects
  seen for less than 2 s are left out; areas the lidar never saw are unknown, not free.
- **Label noise.** Boxes are hand-labelled and loose, which is why pedestrians are evaluated as discs.
- **Poses.** Sequences 8, 14 and 15 only have odometry-quality poses (locally consistent, which is
  what a 2-10 s scenario needs). CODa sequences 21 and 22 have no labels and are not included.
- **Short horizon at 10 Hz**, as noted above.

## Source datasets that were not converted

| Dataset | Status | Reason |
|---|---|---|
| JRDB | not yet | Convertible: it has 3D pedestrian tracks at 15 Hz. Download needs a registered account, which was not available yet; robot odometry is only in the rosbags. It labels pedestrians only (no static objects), and the robot is stationary in 13 of its 27 training sequences. A draft converter is in the code repository, untested on real labels. |
| SCAND | no | No object labels of any kind: the authors state they provide no annotations for human detection or tracking, only coarse per-trajectory tags. Future positions of moving obstacles would have to come from an automatic detector and tracker, which is not ground truth to score collisions against. |
| MuSoHu | no | Same problem: no boxes or tracks, only per-trajectory tags. It is also recorded from a helmet worn by a walking person, so "ego" motion includes head motion. |

SCAND and MuSoHu could still yield ego trajectories with a lidar static map (static-collision
evaluation only); that is not part of this release.

## License and attribution

This dataset is adapted from CODa and is released under the same license,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial, share-alike).
Changes from the original: labels were re-sampled into fixed-length ego-centric windows, tracks were
cleaned and re-identified, velocities and stationarity flags were derived, and obstacle maps were
computed from the lidar sweeps. If you use this data, cite CODa:

```bibtex
@misc{zhang2023robust,
  title={Towards Robust Robot 3D Perception in Urban Environments: The UT Campus Object Dataset},
  author={Arthur Zhang and Chaitanya Eranki and Christina Zhang and Ji-Hwan Park and Raymond Hong and Pranav Kalyani and Lochana Kalyanaraman and Arsh Gamare and Arnav Bagad and Maria Esteva and Joydeep Biswas},
  year={2023},
  eprint={2309.13549},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

Dataset DOI: [10.18738/T8/BBOQMV](https://doi.org/10.18738/T8/BBOQMV).
