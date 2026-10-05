---
license: cc-by-nc-4.0
pretty_name: Robot Navigation Open Scenarios (Habitat HSSD point-goal)
task_categories:
- robotics
tags:
- navigation
- trajectory-prediction
- point-goal
- humanoid
- collision-avoidance
- simulation
- habitat
- camera
- depth
size_categories:
- 10K<n<100K
configs:
- config_name: hssd_2hz
  default: true
  data_files:
  - split: train
    path: data/hssd_2hz/train-*.parquet
  - split: validation
    path: data/hssd_2hz/validation-*.parquet
  - split: test
    path: data/hssd_2hz/test-*.parquet
- config_name: frames
  data_files:
  - split: train
    path: data/frames/train-*.parquet
- config_name: episodes
  data_files:
  - split: train
    path: data/episodes/train-*.parquet
---

# Robot Navigation Open Scenarios: Habitat HSSD point-goal

Image-in, trajectory-out navigation scenarios with 3D ground truth, **generated in simulation** with
[Habitat-Sim](https://aihabitat.org/) in the houses of
[HSSD](https://3dlg-hcvc.github.io/hssd/) (Habitat Synthetic Scenes Dataset, Khanna et al. 2023).
Same schema and evaluator as the real-robot conversions
([CODa](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset),
[RoboSense](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_robosense),
[JRDB](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_jrdb)); the CODa card documents
every column, the coordinate frame, the point and map formats and the evaluation protocol. This set is
the large *training* counterpart of those small real-world test sets. Code:
<https://github.com/naomili0924/qwen_robotics_open_dataset>.

Each scenario has two halves:

- **Input, the past.** Rendered forward-camera images (640 x 480, 90° horizontal field of view, 1.2 m
  above the floor) of the 10 past steps and the current step, the agent's own past positions, and the
  goal: the end of the episode, in the current frame.
- **Ground truth, the future.** The shortest navmesh path the agent follows over the 10 future steps, and
  for the current and each future step the 3D geometry around it: a 360° point cloud from four depth
  cameras, every point tagged ground / static, plus the merged static obstacle map. There are no other
  agents (`future_tracks` is empty, `num_pedestrians` is 0).

![Example scenarios](figures/hssd_2hz_examples.png)

*Three test scenarios. Left: the current image with the future path drawn on the floor. Middle:
bird's-eye view with the static map (dark = occupied), ego in green, goal as a star. Right: the points of
the last future step.*

<!-- per-frame:begin -->
## Per-frame training configs: `frames` and `episodes`

The same recordings in the per-frame training format of
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset) (described in full on the
[EgoWalk card](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_egowalk)): one row per
camera frame with the raw metric pose (x, y, z, yaw), plus one row per episode with camera, environment and
embodiment. Training samples (history, waypoints by distance along the path, a goal beyond the horizon, a text
prompt) are cut at load time with `hnod.windows.FrameWindows`.

| Split | Episodes | Frames | Rate | Hours | km | Indoor frames | Median speed |
|---|---|---|---|---|---|---|---|
| train | 7,505 | 158,855 | 2 Hz | 21.0 | 75.5 | 91% | 1.00 m/s |

Built from `hssd_2hz` train: frames at 2 Hz rendered at 1.0 m/s by a shortest-path agent; episodes end at their goal (`ends_at_rest`). Some episodes leave the houses through the gardens (the generator's room-polygon restriction leaks): use `frame_indoor_prob` or `environment` to keep indoor ones. Frames were collected from the scenario rows (each imaged frame once) and their poses mapped back to the
source's world frame; the last second or so of each recorded segment, which has poses but no images in the
scenario rows, is dropped. Indoor / outdoor: `estimated:clip-vit-l14`, a rough estimate: CLIP tends to call courtyards and
covered walkways indoor (the evaluation suite's labels were reviewed by eye instead). **Only the train split is published here: the
held-out splits of this source are part of the evaluation suite
([qwen_robotics_nav_eval](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_nav_eval)) and must not be
trained or tuned on.**

```python
from datasets import load_dataset
from hnod.windows import FrameWindows
frames = load_dataset("Jinyan0924/habitat_hssd_pointgoal_nav_scenarios", "frames", split="train")
episodes = load_dataset("Jinyan0924/habitat_hssd_pointgoal_nav_scenarios", "episodes", split="train")
samples = FrameWindows(frames, episodes)
```
<!-- per-frame:end -->

## What is in it

| Config | Step | Past + future | Scenario spacing | Scenes (train / validation / test) | Scenarios (train / validation / test) |
|---|---|---|---|---|---|
| `hssd_2hz` | 0.5 s | 5 s + 5 s | 1 s | SCENES_PLACEHOLDER | ROWS_PLACEHOLDER |

The agent walks at 1 m/s along the geodesic shortest path between a random navigable start and goal
12–40 m apart, with the heading of travel; a 40-step episode yields about 15 overlapping scenarios. Splits
are by scene, following HSSD's own `scene_splits.yaml`: its `val` scenes are divided into our validation
and test sets, its `train` scenes are our train set. Scenario ids are
`hssd_<scene>_<seed>_<episode>_<step>_2hz`.

```python
from datasets import load_dataset
ds = load_dataset("Jinyan0924/habitat_hssd_pointgoal_nav_scenarios", "hssd_2hz", split="test")
```

```bash
python scripts/evaluate.py --repo Jinyan0924/habitat_hssd_pointgoal_nav_scenarios --config hssd_2hz \
    --split test --pred my_predictions.json        # {"<scenario_id>": [[x, y] * 10], ...}
```

## Differences from the real-robot sets

- **Agent.** A nominal humanoid, 0.6 x 0.6 m footprint, 1.7 m tall (`ego` size). The navmesh is
  recomputed for that body (radius 0.3 m, height 1.7 m, scene objects included), so the recorded path
  keeps the evaluator's 0.3 m clearance from furniture and walls by construction.
- **Images** are rendered (Habitat PBR renderer, default lighting), not photographed; there is no
  distortion, `camera.K` is the pinhole model of the 90° camera.
- **Lidar** is replaced by four rendered depth cameras (forward / left / back / right, 25 m range) fused
  into a 360° point cloud; the same thinning and labelling as for real lidar is applied. Labels are
  ground (0) or static (1); there is nothing dynamic.
- **Motion** is piecewise straight at constant speed, with instantaneous turns at path corners: a
  planner's reference path rather than a human-like walk. `ego` velocities are finite differences of it.
- **No other agents.** Collision-avoidance in this set is purely against static structure.

## How it was built

`scripts/generate_habitat_pointnav.py` (habitat-sim 0.3.1, headless EGL rendering). For each scene:
load it through HSSD's `hssd-hab.scene_dataset_config.json`, recompute the navmesh, sample episodes,
step the agent along the resampled path at 2 Hz, render, convert depth to points in the ego frame, and
run the same windowing / map / point code as the real datasets (`hnod.pipeline.segment_rows`).

## Baselines

Test split, collision rates in %, ADE/FDE in m, default evaluator settings (agent radius 0.3 m).

BASELINES_PLACEHOLDER

## Limitations

- **Synthetic.** Rendered images and clean geometry; expect a domain gap to the real-robot sets.
- **Static world.** No people or moving objects.
- **Open loop, reference-path motion.** The ground truth is a shortest path, not a demonstration.
- **One camera.** The past contains only the forward view; the 360° depth is ground truth, not input.

## License and attribution

HSSD is published under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); this derived
dataset carries the same license (non-commercial). Rendered images and geometry are derived works of the
HSSD scenes and object models. If you use this data, cite HSSD and Habitat:

```bibtex
@inproceedings{khanna2023hssd,
  title={Habitat Synthetic Scenes Dataset (HSSD-200): An Analysis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal Navigation},
  author={Khanna, Mukul and Mao, Yongsen and Jiang, Hanxiao and Haresh, Sanjay and Shacklett, Brennan and Batra, Dhruv and Clegg, Alexander and Undersander, Eric and Chang, Angel X. and Savva, Manolis},
  booktitle={CVPR},
  year={2024}
}
@inproceedings{szot2021habitat,
  title={Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
  author={Szot, Andrew and Clegg, Alexander and Undersander, Eric and Wijmans, Erik and Zhao, Yili and Turner, John and Maestre, Noah and Mukadam, Mustafa and Chaplot, Devendra and Maksymets, Oleksandr and Gokaslan, Aaron and Vondrus, Vladimir and Dharur, Sameer and Meier, Franziska and Galuba, Wojciech and Chang, Angel and Kira, Zsolt and Koltun, Vladlen and Malik, Jitendra and Savva, Manolis and Batra, Dhruv},
  booktitle={NeurIPS},
  year={2021}
}
```
