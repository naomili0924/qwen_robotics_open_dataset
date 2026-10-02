---
license: cc-by-nc-sa-4.0
pretty_name: Robot Navigation Open Scenarios (RoboSense)
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
- 10K<n<100K
configs:
- config_name: robosense_1hz
  default: true
  data_files:
  - split: train
    path: data/robosense_1hz/train-*.parquet
  - split: validation
    path: data/robosense_1hz/validation-*.parquet
---

# Robot Navigation Open Scenarios: RoboSense

Waymo-Open-Motion-style scenarios for evaluating a navigating robot (the intended use is
humanoid navigation), converted from [RoboSense](https://github.com/suhaisheng/RoboSense)
(Su et al., CVPR 2025). Each scenario gives **10 past steps, the current step and 10 future
steps** of the ego robot and of the tracked vehicles, pedestrians and cyclists around it, plus
a static obstacle map. A planner predicts the ego's 10 future positions and is scored on
whether that path collides with moving agents, stationary objects or static structure.

Same schema and evaluator as the CODa conversion,
[Jinyan0924/qwen_robotics_open_dataset](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset),
whose card documents every column, the coordinate frame and the map image format; this card
only lists what is different. Code: <https://github.com/naomili0924/qwen_robotics_open_dataset>.

![Example scenarios](figures/robosense_1hz_examples.png)

*Six validation scenarios. Green: ego past (faint) and future (dotted), star = goal. Red: pedestrians,
orange: cyclists, purple: vehicles, grey box: likely operator. Background: static map
(dark = occupied, white = free, grey = unknown).*

## What is in it

| Config | Step | History + future | train | validation |
|---|---|---|---|---|
| `robosense_1hz` | 1 s | 10 s + 10 s | 12,754 | 1,039 |

RoboSense is labelled at 1 Hz, so 10 past and 10 future frames span 10 s each way. Consecutive
training scenarios are 2 s apart, validation scenarios 1 s apart. Splits follow RoboSense's own
train / val files; its test split has no public labels.

```python
from datasets import load_dataset
ds = load_dataset("Jinyan0924/qwen_robotics_open_dataset_robosense", "robosense_1hz", split="validation")
```

```bash
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset_robosense --config robosense_1hz \
    --split validation --pred my_predictions.json        # {"<scenario_id>": [[x, y] * 10], ...}
```

## Differences from the CODa conversion

- **Platform.** A small road-sweeper robot driven by remote control below about 1 m/s through parks,
  squares, campuses, streets and sidewalks. Its size is not published; `ego.length/width/height` is a
  nominal 1.5 x 0.9 x 1.4 m. The ego reference point is the rear axle.
- **Object types.** Only `VEHICLE`, `PEDESTRIAN` and `CYCLE` are labelled. There are no `STATIC` tracks:
  poles, trees, walls and kerbs exist only in the static map.
- **Scenario ids.** `robosense_<first clip token>_<frame>_1hz`. `sequence` is the token of the first
  source clip in the chain, `source_frames` counts frames (seconds) from the start of that clip.
- **Occlusion** labels do not exist; `occlusion` is 5 (unknown) wherever a track is valid.
- **Stationary threshold.** A track counts as stationary if it stays within 0.5 m of its median position
  (0.25 m for CODa), because poses and labels jitter more here.
- **Map.** Built from the 64-beam Hesai lidar, merging up to 11 sweeps (5 s either side of the current
  step) for `static_map` and up to 6 (the past 5 s) for `observed_map`. An obstacle cell must be seen
  occupied at least 2 s apart to count as static.

## How it was built

1. **Clips to chains.** RoboSense cuts its recordings into clips of about 20 s, too short for a 21-step
   window at 1 Hz. Clips that follow each other without a gap inside the same split are chained: 864
   chains of at least 21 frames in train, 61 in validation. Chains are also cut at the 174 places
   where the recorded ego pose jumps faster than 3 m/s.
2. **Tracks.** The source ids are not clean track ids: they restart in every clip, 6.5% of frames contain
   the same id twice, and 2.5% of id links imply impossible speeds. Tracks are therefore rebuilt by
   frame-to-frame association on position with a constant-velocity prediction; a source id is used only
   when the motion it implies is physically possible.
3. **Boxes.** Converted from bottom-centre to centre; headings are disambiguated front/back by the
   direction of travel for objects that move.
4. **Static map.** Same ground/obstacle segmentation as for CODa, run on 44,313 lidar sweeps streamed
   from the source archive. A further 556 sweeps (1.2%) are missing from that archive; the maps around
   them use the neighbouring sweeps only.

## Baselines

Collision rates in %, ADE/FDE in m, default evaluator settings (agent radius 0.3 m):

| validation | any | dynamic | static | map | ADE | FDE |
|---|---|---|---|---|---|---|
| recorded path (`expert`) | 2.6 | 0.3 | 0.4 | 1.9 | 0.00 | 0.00 |
| `straight_to_goal` | 6.7 | 0.7 | 0.9 | 5.4 | 0.36 | 0.00 |
| `constant_velocity` | 18.0 | 9.5 | 1.5 | 8.4 | 0.96 | 2.04 |
| `stationary` | 35.6 | 35.6 | 0.0 | 0.0 | 3.94 | 7.14 |

The recorded path did not hit anything, so its 2.6% (3.9% on train) is the label-noise floor here,
about three times CODa's. `stationary` scores badly because other agents replay their recorded motion:
people and vehicles that followed the robot walk through the spot where it would have stopped.

## Limitations

- **1 Hz labels.** Agent positions between steps are linear interpolations over a full second, which is
  coarse for vehicles and cyclists. Collisions with fast agents between steps can be missed or invented.
- **Track identity is reconstructed.** In dense crowds the association can hop between neighbours a metre
  or two apart (visible as zig-zags in the first example). Box positions at each labelled step are
  unaffected; interpolated positions between steps and velocities for such tracks are not reliable.
  Tracks are short: the median track lasts 3 steps.
- **Noisier geometry than CODa.** Parked cars wander by a few decimetres in the world frame over a
  chain, so walls in the merged map are slightly thickened.
- **Many near-stationary scenarios.** In 19% of training and 10% of validation scenarios the robot moves
  less than 1 m in the future window (`ego_future_distance`); filter on that column if you only want
  scenarios with motion.
- **Unknown space.** The lidar cannot see the ground within about 3 m of the robot and its rings are
  sparse, so 22% of the recorded path in validation (31% in train) lies in cells marked unknown (not occupied).
- **Not a humanoid recording, open loop, operator heuristic, derived static map:** as for CODa. The
  operator flag has not been checked against ground truth for this platform.
- **No terrain labels** exist to verify the ground segmentation on this sensor; it was checked only
  through the recorded path's collision rate and by eye.

## License and attribution

Adapted from RoboSense. The RoboSense authors state CC BY-NC-SA 4.0 in their README and dataset card
(the Hugging Face metadata field of their repository says CC BY-SA 4.0); this derived dataset is released
under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), the stricter of the two.
Changes from the original: clips were chained, tracks re-associated, boxes re-expressed in ego-centric
windows, velocities and stationarity flags derived, and obstacle maps computed from the lidar sweeps.
If you use this data, cite RoboSense:

```bibtex
@inproceedings{su2025robosense,
  title={RoboSense: Large-scale Dataset and Benchmark for Egocentric Robot Perception and Navigation in Crowded and Unstructured Environments},
  author={Su, Haisheng and Song, Feixiang and Ma, Cong and Wu, Wei and Yan, Junchi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
