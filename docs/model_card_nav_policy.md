---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-VL-3B-Instruct
pipeline_tag: robotics
tags:
- navigation
- vision-language-action
- lora
- indoor
- humanoid
datasets:
- Jinyan0924/qwen_robotics_open_dataset_egowalk
- Jinyan0924/habitat_hssd_pointgoal_nav_scenarios
- Jinyan0924/qwen_robotics_open_dataset_robosense
- Jinyan0924/qwen_robotics_open_dataset
---

# Qwen robotics navigation policy (work in progress)

An image-in, path-out navigation policy for a slow walking robot: it looks at the last few front-camera
images, reads a prompt that contains the goal ("Please walk towards the goal (4.0, -1.0)."), and returns the
next 2 m of path as 8 waypoints. It is Qwen2.5-VL-3B with LoRA adapters and a small action head, trained by
imitation on open navigation data. Code, data conversion and evaluation:
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset).

> **Status: an unfinished first run, published for transparency, not a finished model.** Run `e1_all_sqrt` is
> at about step 8,000 of 69,400 (one pass over the data); the checkpoints here update as training continues.
> It collides often in cluttered houses (see Results). **Research use only, non-commercial** (see Licence).
> Not safety-tested: do not run it on a robot near people.

## Checkpoints

All under `e1_all_sqrt/`:

| Folder | What |
|---|---|
| `best/` | lowest validation loss so far (`best.json` gives the step); validation = held-out EgoWalk recordings, not the evaluation suite |
| `step_5000/`, `step_10000/`, ... | a fixed checkpoint every 5,000 steps |
| `last/` | newest checkpoint (every 1,000 steps), with optimiser state for resuming; older versions are in the commit history |
| `eval/` | predictions and metrics on the evaluation suite; `log.jsonl` is the training log |

Each checkpoint is `lora/` (adapter for the language model), `head.pt` (action head) and `config.json`. They
are loaded by the project's code, not by `peft` alone.

## Use

```bash
git clone https://github.com/naomili0924/qwen_robotics_open_dataset && cd qwen_robotics_open_dataset
pip install -r requirements-train.txt

# your own images (oldest first, 1 s apart, the last is the current view) and a goal in the robot frame
python -m vla.infer --hub-repo Jinyan0924/qwen_robotics_nav_policy --checkpoint e1_all_sqrt/best \
    --images t-5.jpg t-4.jpg t-3.jpg t-2.jpg t-1.jpg now.jpg --goal 4.0 -1.0 --plot path.png

# score a checkpoint on the evaluation suite
python -m vla.predict_suite --hub-repo Jinyan0924/qwen_robotics_nav_policy --checkpoint e1_all_sqrt/best \
    --version v2 --out best_v2.json
```

```python
from vla.pipeline import NavigationPipeline
pipe = NavigationPipeline.from_hub("Jinyan0924/qwen_robotics_nav_policy", "e1_all_sqrt/best")
path = pipe.predict_path(images, goal=(4.0, -1.0))    # (8, 2) waypoints in metres, 0.25 m apart
```

- **Inputs:** up to 6 front-camera images 1 s apart (resized to about 336 x 336 pixels of area), optionally the
  robot's own past positions, and a goal (x, y) in metres in the robot frame (robot at (0, 0), x forward, y left)
  and / or a text instruction ("Walk to the glass door on the left.").
- **Output:** 8 waypoints (x, y) spaced 0.25 m along the path. Spacing is by distance, not time, so the path does
  not prescribe a speed.
- Needs a GPU with about 16 GB of memory.

## Training

| | |
|---|---|
| Base model | Qwen/Qwen2.5-VL-3B-Instruct; vision encoder frozen |
| Trained parts | LoRA rank 16 (alpha 32) on the language model's attention and MLP projections, 29.9 M parameters; regression action head, 9.7 M parameters |
| Objective | imitation: regress the recorded path (8 waypoints) from images, past positions and the prompt |
| Data | 1.11 M frames, 79 h: EgoWalk (people walking, 905 k frames), HSSD simulated houses (159 k), RoboSense (26 k), CODa (20 k); train splits only |
| Sampling | sources drawn in proportion to the square root of their size (about 58 / 24 / 10 / 8 %) |
| Prompts | point goals 4-20 m ahead on the recorded path; for EgoWalk also its language goals |
| Optimisation | AdamW, batch 16, learning rate 1e-4 (LoRA) / 3e-4 (head), 500 warm-up steps, cosine decay over 69,400 steps |
| Hardware | one H100, about 4.8 s per step |

## Results so far

[Evaluation suite](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_nav_eval): 150 audited scenarios;
the predicted path is followed at 0.5 m/s and scored for at-fault collisions and progress towards the goal.
Success = no collision and at least half of the possible progress. Checkpoint `step_5000`:

| Scenarios | Success | Collision | "Straight to goal" success / collision |
|---|---|---|---|
| v2 indoor, 100 real | 80% | 9% | 69% / 31% |
| v1 indoor, 25 real + 75 simulated houses | 35% | 58% | 20% / 80% |
| of which simulated houses (75) | 17% | 77% | 0% / 100% |
| outdoor, 50 real | 26% | 4% | 68% / 32% |

- Real indoor scenes (corridors, halls, a mall): better than naive planners, mainly by colliding less.
- Simulated houses with doorways and furniture: it still collides in most scenarios.
- **The outdoor number is a scoring artefact, not model behaviour:** the model predicts 2 m of path, while the
  33 RoboSense scenarios are scored over 10 s (5 m at 0.5 m/s), so it cannot reach the progress threshold there.
- Scores of three early checkpoints (steps 1,000 / 3,000 / 5,000) moved by about 10 points without a trend; with
  100 scenarios that is within noise. No conclusion about the final model should be drawn from them.

## Limitations

- Unfinished training; open-loop evaluation only (recorded people do not react to the robot); never run on a robot.
- Imitation only: nothing in training penalises a collision.
- Most training data is a person walking at about 1.2 m/s with a chest-height camera; the target robot is slower.
- Predicts 2 m ahead and no heading; it does not decide when to stop.

## Licence

Research use only, non-commercial. The base model is under the
[Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE); read it before
using these weights. The training data adds non-commercial terms: HSSD (CC BY-NC 4.0), RoboSense and UT CODa
(CC BY-NC-SA 4.0); EgoWalk is MIT. Cite Qwen2.5-VL and the source datasets.
