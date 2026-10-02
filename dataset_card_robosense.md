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
- camera
size_categories:
- 1K<n<10K
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

Image-in, trajectory-out navigation scenarios with 3D ground truth, for evaluating a navigating
robot (the intended use is humanoid navigation). Converted from
[RoboSense](https://github.com/suhaisheng/RoboSense) (Su et al., CVPR 2025).

Each scenario has two halves:

- **Input, the past.** Front-camera images of the 10 past frames and the current frame, plus the
  robot's own past positions. No obstacle information.
- **Ground truth, the future.** The robot's recorded trajectory over the 10 future frames, and for the
  current and each future frame the 3D geometry around it: boxes of every labelled vehicle, pedestrian
  and cyclist (static or dynamic) and a lidar point cloud with every point tagged
  ground / static / dynamic.

Same schema and evaluator as the CODa conversion,
[Jinyan0924/qwen_robotics_open_dataset](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset),
whose card documents every column, the coordinate frame, the point and map formats and the evaluation
protocol; this card lists what is different. Code: <https://github.com/naomili0924/qwen_robotics_open_dataset>.

![Example scenarios](figures/robosense_1hz_examples.png)

*Three validation scenarios. Left: the current image with the recorded future path drawn on the ground.
Middle: bird's-eye view with the static map (dark = occupied), boxes and their future tracks (red
pedestrians, orange cyclists, purple vehicles, grey likely operator), ego in green, goal as a star.
Right: the lidar points of the last future step (black static, red dynamic).*

## What is in it

| Config | Step | Past + future | train | validation |
|---|---|---|---|---|
| `robosense_1hz` | 1 s | 10 s + 10 s | 5,869 | 963 |

RoboSense is labelled at 1 Hz, so 10 past and 10 future frames span 10 s each way. Consecutive
training scenarios are 4 s apart, validation scenarios 1 s apart. Splits follow RoboSense's own
train / val files; its test split has no public labels. A scenario is about 9 MB; use streaming.

```python
from datasets import load_dataset
ds = load_dataset("Jinyan0924/qwen_robotics_open_dataset_robosense", "robosense_1hz", split="validation", streaming=True)
row = next(iter(ds))
images = row["past_images"]          # 11 PIL images, oldest first, last = current
```

```bash
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset_robosense --config robosense_1hz \
    --split validation --pred my_predictions.json        # {"<scenario_id>": [[x, y] * 10], ...}
```

## Differences from the CODa conversion

- **Platform.** A small road-sweeper robot driven by remote control below about 1 m/s through parks,
  squares, campuses, streets and sidewalks. Its size is not published; `ego.length/width/height` is a
  nominal 1.5 x 0.9 x 1.4 m. The ego reference point is the rear axle.
- **Images.** `CAM_FRONT`, the forward pinhole camera, 1920 x 1080, as the original JPEG files. They are
  already undistorted, so `camera.K` alone describes them. The camera sits 0.7 m ahead of the rear axle
  and 0.74 m above it.
- **Object types.** Only `VEHICLE`, `PEDESTRIAN` and `CYCLE` are labelled. There are no `STATIC` boxes:
  poles, trees, walls and kerbs exist only as lidar points and in the static map.
- **Scenario ids.** `robosense_<first clip token>_<frame>_1hz`. `sequence` is the token of the first
  source clip in the chain, `source_frames` counts frames (seconds) from the start of that clip.
- **Occlusion** labels do not exist; `occlusion` is 5 (unknown) wherever a box is valid.
- **Stationary threshold.** An object counts as stationary if it stays within 0.5 m of its median
  position (0.25 m for CODa), because poses and labels jitter more here.
- **Lidar.** The 64-beam Hesai sweep of each step. `static_map` merges up to 11 sweeps (5 s either side
  of the current step); a cell must be seen occupied at least 2 s apart to count as static.

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
4. **Lidar.** Same ground/obstacle split as for CODa, on sweeps streamed from the source archive. One
   recording batch stores its sweeps in the vehicle frame instead of the sensor frame, which is handled.
5. **Missing files.** 1,487 of the 44,869 front images needed (3.3%) and 556 of the lidar sweeps (1.2%) are absent from the source archives. Scenarios that need one of them are left out.

## Baselines

Validation split, collision rates in %, ADE/FDE in m, default evaluator settings (agent radius 0.3 m).
`any` = dynamic or static or map; `lidar` is the independent per-step point check.

| | any | dynamic | static | map | lidar | ADE | FDE |
|---|---|---|---|---|---|---|---|
| recorded path (`expert`) | 2.9 | 0.3 | 0.5 | 2.1 | 3.0 | 0.00 | 0.00 |
| `straight_to_goal` | 7.4 | 0.6 | 1.1 | 5.8 | 7.8 | 0.37 | 0.00 |
| `constant_velocity` | 19.3 | 10.2 | 1.7 | 9.0 | 20.6 | 1.00 | 2.13 |
| `stationary` | 35.9 | 35.9 | 0.0 | 0.0 | 27.3 | 3.90 | 7.08 |

The recorded path did not hit anything, so its rates are the label-noise floor here, about three
times CODa's (on train: any 3.7, lidar 1.7). `stationary` scores badly because other agents replay their recorded motion: people and
vehicles that followed the robot pass through the spot where it would have stopped.

## Limitations

- **1 Hz labels.** Agent positions between steps are linear interpolations over a full second, which is
  coarse for vehicles and cyclists. Collisions with fast agents between steps can be missed or invented.
  The lidar check sees each place only at whole seconds.
- **Track identity is reconstructed.** In dense crowds the association can hop between neighbours a metre
  or two apart (visible as zig-zags in the first example). Box positions at each labelled step are
  unaffected; interpolated positions between steps and velocities for such tracks are not reliable.
  Tracks are short: the median track lasts 3 steps.
- **Noisier geometry than CODa.** Parked cars wander by a few decimetres in the world frame over a
  chain, so walls in the merged map are slightly thickened.
- **Many near-stationary scenarios.** In 20% of training and 10% of validation scenarios
  the robot moves less than 1 m in the future window (`ego_future_distance`); filter on that column if
  you only want scenarios with motion.
- **Unknown space.** The lidar cannot see the ground within about 3 m of the robot and its rings are
  sparse, so 22% of the recorded path in validation lies in map cells marked unknown (not occupied).
- **One forward camera, not a humanoid recording, open loop, operator heuristic, derived geometry:** as
  for CODa. The operator flag has not been checked against ground truth for this platform.
- **No terrain labels** exist to verify the ground segmentation on this sensor; it was checked only
  through the recorded path's collision rate and by eye.

## License and attribution

Adapted from RoboSense. The RoboSense authors state CC BY-NC-SA 4.0 in their README and dataset card
(the Hugging Face metadata field of their repository says CC BY-SA 4.0); this derived dataset is released
under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), the stricter of the two.
Changes from the original: clips were chained, tracks re-associated, boxes re-expressed in ego-centric
windows, velocities and static/dynamic flags derived, lidar sweeps thinned, labelled and merged into
obstacle maps. Images are unmodified. If you use this data, cite RoboSense:

```bibtex
@inproceedings{su2025robosense,
  title={RoboSense: Large-scale Dataset and Benchmark for Egocentric Robot Perception and Navigation in Crowded and Unstructured Environments},
  author={Su, Haisheng and Song, Feixiang and Ma, Cong and Wu, Wei and Yan, Junchi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
