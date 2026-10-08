---
license: other
license_name: mixed-non-commercial
task_categories:
- robotics
tags:
- navigation
- egocentric
- trajectory-prediction
- deduplicated
pretty_name: Navigation pretraining frames, deduplicated
size_categories:
- 100K<n<1M
configs:
- config_name: frames
  default: true
  data_files:
  - split: train
    path: data/frames/train-*.parquet
- config_name: episodes
  data_files:
  - split: train
    path: data/episodes/train-*.parquet
- config_name: samples
  data_files:
  - split: train
    path: data/samples/train-*.parquet
- config_name: annotations
  data_files:
  - split: train
    path: data/annotations/train-*.parquet
---

# Navigation pretraining frames, deduplicated

A **49,391-sample** training set for the final-frame pretraining task of
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset): past camera views + the view
5 s ahead → the recorded motion in between. It is cut from the same four open sources as the full per-frame sets
(1.11 M frames), keeping **motion diversity first and scene diversity second**, so that a pass over it costs a
fraction of the time and the rare behaviours (turns, stops, starts) are not drowned by straight walking.

![motion classes](figures/buckets.png)

![examples](figures/examples.png)

## How it was built

1. **Candidates:** one sample per second of every training recording (`scripts/build_dedup_index.py`): the next
   5 s of recorded motion is summarised (mean speed, speed at the start and end, net heading change, straightness)
   and the current view is embedded with CLIP ViT-L/14.
2. **Motion class** of each candidate: straight, standing, slowing, speeding_up, turn_left, turn_right, sharp_left, sharp_right, stopping, starting, weaving (`bucket()` in the same script).
3. **Selection** (`scripts/select_dedup.py`), per source and motion class: rare classes are kept up to
   6,000 each, `straight` is capped at 6,000 and `standing` at
   1,500; inside a class, candidates are visited in random order and one is skipped when its
   scene embedding is more than 0.90 cosine-similar to a kept sample of the same class (same place,
   same view, same motion = duplicate). Simulated houses: at most 120 samples per house.
4. **Publication** (`scripts/publish_dedup.py`): the per-frame format of the full sets, restricted to episodes with
   kept samples; images (long side 640 px) only for the frames a kept sample needs.

| Source | Candidates | Kept | Share |
|---|---|---|---|
| HSSD | 45,587 | 9,556 | 21% |
| CODa | 1,719 | 283 | 16% |
| EgoWalk | 177,647 | 35,480 | 20% |
| RoboSense | 21,943 | 4,072 | 19% |
| **Total** | **246,896** | **49,391** | **20%** |

Kept samples per motion class:

| Class | HSSD | CODa | EgoWalk | RoboSense | Total |
|---|---|---|---|---|---|
| straight | 3,336 | 147 | 6,000 | 1,293 | 10,776 |
| standing | 0 | 14 | 634 | 262 | 910 |
| slowing | 0 | 17 | 2,225 | 488 | 2,730 |
| speeding_up | 0 | 9 | 1,782 | 503 | 2,294 |
| turn_left | 1,956 | 29 | 6,000 | 312 | 8,297 |
| turn_right | 1,770 | 26 | 6,000 | 301 | 8,097 |
| sharp_left | 1,273 | 8 | 4,149 | 319 | 5,749 |
| sharp_right | 987 | 6 | 6,000 | 83 | 7,076 |
| stopping | 0 | 12 | 648 | 274 | 934 |
| starting | 0 | 13 | 608 | 204 | 825 |
| weaving | 234 | 2 | 1,434 | 33 | 1,703 |

![scene map](figures/scene_map.png)

*Scene embeddings projected on two axes: the kept samples cover the same regions as the candidates with far fewer
points; the simulated houses (orange) collapse from a dense blob to a thin layer.*

## Contents

- `frames` (1,055,580 rows, 158,838 with an image): `episode_id`, `frame_index`, `source_frame`,
  `timestamp`, `image` (None where no kept sample needs it), `image_right` (None), `pose` (x, y, z, yaw).
- `episodes` (6,003 rows): the source episode tables (embodiment, rate, camera, environment, ...).
- `samples` (49,391 rows): `episode_id`, `frame_index`, `source`, `embodiment`, `bucket`, the motion
  summary of the next 5 s, and the text the model sees: `system` (fixed) and `prompt` (rendered from the
  embodiment: *"The camera is carried by a human walking. The past views are sampled at 1 Hz. Please predict
  the next 10 positions at 2 Hz."*). The task gives no goal: the final frame is the goal. `camera_prompt` is the
  optional camera sentence (`--camera-prompt`): field of view from the stored intrinsics and the height above the
  ground **only when its source is the dataset's calibration or the simulator** (`episodes.camera.height_source`);
  anything estimated or missing is written as "unknown". The calibration and its provenance are also kept in the code
  repository (`docs/calibration/`).
- `annotations` (one row per sample): a description written by Claude Haiku 4.5 from 11 frames (5 s before to
  5 s after the current frame, 1 Hz) and the recorded path: `description` (one or two sentences on where the
  carrier goes, what it passes, avoids or waits for, whether it stops or turns), `place`, `interaction`, plus the
  sample's `camera_prompt` (the same sentence as in `samples`). The text is hindsight: used as an auxiliary training
  target (`--text-loss`) or, for the instruction-conditioned policy, as the task text in the prompt
  (`--instruction description,place,interaction`), in which case the score measures text following
  (`scripts/annotate_samples.py`).

## Descriptions (`annotations`)

All 49,391 samples were described by Claude Haiku 4.5 through the Batch API (158 M input tokens,
3.9 M output tokens, about $89). The model saw 11 frames at 1 Hz (5 s before the current frame
to 5 s after) and the recorded path of the next 5 s as text, and answered with a `description`, a `place` and an
`interaction` (what the carrier reacts to; "none" for 23% of the samples). Mean length
34 words. One example per motion class:

- *sharp_left* (egowalk): The carrier walks at a steady slow pace through a furniture showroom, passing display beds and cabinets while bearing left to navigate around a large wardrobe display that occupies the center of the space.
- *sharp_right* (egowalk): The carrier walks at a slow pace through a shopping mall corridor, passing by a decorative white Christmas tree and a railing on the left, then gradually turns right toward the food court area with the Stardogs restaurant kiosk.
- *slowing* (egowalk): The carrier walks straight ahead at a slow, steady pace through a wet parking lot in front of apartment buildings, maintaining their course without deviation or interaction with any obstacles or people.
- *speeding_up* (egowalk): The carrier walks at a steady, slightly increasing pace straight ahead through a shopping mall corridor, passing by store fronts and avoiding scattered pedestrians while maintaining a direct path forward.
- *standing* (robosense): The carrier moves slowly straight ahead at walking pace along an urban street with storefronts on the right and trees lining the left side, then comes to a stop after traveling a short distance.
- *starting* (egowalk): The carrier walks at a slow pace across a large open plaza area and then turns sharply left toward the building structures along the left side. The movement suggests the carrier is navigating through an outdoor commercial or transportation hub and redirecting toward the storefronts and covered areas visible on the left.
- *stopping* (egowalk): The carrier walks at a slow pace through an interior space, then stops and turns around to the left, likely reacting to reaching a destination or realizing a need to change direction in the hallway or entryway.
- *straight* (egowalk): The carrier walks straight ahead at a steady walking pace along a paved pathway beside a green fence in an urban nighttime setting, passing by street lights and commercial signage on the left side.
- *turn_left* (egowalk): The carrier walks at a slow pace through a flower shop, bearing slightly left while exiting through the glass doors toward the street outside. The carrier navigates around a metal basket display stand and heads toward the entrance, ultimately passing through the doors to reach the exterior.
- *turn_right* (egowalk): The pedestrian walks at a steady pace across a wide urban plaza, gradually bearing right to navigate around parked vehicles while maintaining forward progress through the open square.
- *weaving* (egowalk): The carrier walks slowly and steadily straight ahead through a furniture showroom, passing by display counters and store fixtures while maintaining a consistent pace in the open floor space.

The text is hindsight (it knows what happened), so it is a training *target* (`--text-loss`), never an input.

## Use

```bash
bash scripts/run_ff.sh <run> <steps> 2b --frames-repos Jinyan0924/qwen_robotics_nav_pretrain_dedup
```

The streaming loader reads the `samples` table and cuts only the listed samples; sources are already balanced
inside the selection, so `--frames-mix` has no effect with a single repo.

## Limits

- Deduplication is by appearance of the current view and by the motion class; two walks through the same corridor
  with the same motion are kept once even if the people around differed.
- The motion classes are thresholds on 5 s of recorded motion (20° and 60° of heading change, 0.15 m/s for
  standing, 0.3 m/s speed change); they describe the recording, not an intent.
- Licences follow the sources: EgoWalk MIT, HSSD CC BY-NC 4.0, RoboSense and CODa CC BY-NC-SA 4.0 (non-commercial).
