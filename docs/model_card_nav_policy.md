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

> **Update 2026-10-06:** run `e1_all_sqrt` was stopped at step 12,000. Tests showed it takes its goal from a
> numeric side-channel and ignores the prompt text, so it is kept only as a baseline. Folders starting with `ff`
> belong to the next stage, final-frame pretraining (past frames + the frame 5 s ahead -> motion), described in
> `docs/final_frame_pretraining.md` of the code repository; `ff0_e2e_test` is a 500-step pipeline test, not a model to use.

## Checkpoints

**Final-frame forecasting model (recommended): `ff3_lora4b_dedup/`** — `Qwen/Qwen3.5-4B-Base` + LoRA (rank 32) on
the language model, action head MLP. Input: 5 past frames (1 Hz), the current frame and the frame 5 s ahead, plus
one sentence saying who carries the camera (human / robot); output: 10 positions at 0.5 s. Trained on the 49,391
motion-deduplicated samples of
[`qwen_robotics_nav_pretrain_dedup`](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_nav_pretrain_dedup)
with their descriptions as an auxiliary text target. Stopped at step 4,325; `step_1000` ... `step_4000`, `best` and
`last` (= step 4,000) are kept, each about 280 MB (`lora/`, `head.pt`, `config.json`) plus `eval/` with the
predictions on both evaluation suites. It needs the future frame, so it is a forecasting / pretraining model, not an
onboard policy. Earlier final-frame runs: `ff0_e2e_test` (Qwen3-VL-2B, pipeline test) and `ff2_qwen35_2b` (2B full
fine-tune on all 1.1 M frames, stopped at step 3,450; it did not read the final frame).

**Goal-prompt baseline: `e1_all_sqrt/`** (Qwen2.5-VL-3B LoRA, goal as text + numbers; the model ignored the goal):

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

### Final-frame task (`ff3_lora4b_dedup/step_4000`)

Evaluation configs `v2_final_frame` (134 scenarios, two thirds straight) and `v3_final_frame` (132 motion-diverse
scenarios) of the suite; the prediction is scored at the recorded timing (collision with people and the map,
completion = end point within 1 m of the recorded end, ADE / FDE to the recorded path).

| | success | completed | collided | ADE (m) | FDE (m) |
|---|---|---|---|---|---|
| v3: constant velocity | 0.33 | 0.38 | 0.24 | 0.84 | 1.92 |
| **v3: ff3 step 4,000** | **0.43** | **0.54** | 0.27 | **0.70** | **1.25** |
| v2: constant velocity | 0.62 | 0.70 | 0.20 | 0.49 | 1.12 |
| v2: ff3 step 4,000 | 0.51 | 0.58 | 0.19 | 0.56 | 1.00 |

Input-use checks on v3: with another scenario's final frame success drops to about 0.10 and the predicted end point
moves 2.6 m; without the final frame to 0.06; with the human / robot sentence swapped to about 0.2. Robot scenarios
(success 0.6, collided 0.1) are easier than walking-person ones (0.33, 0.29); indoor collisions with people are the
weakest point.

### Goal-prompt baseline (`e1_all_sqrt/step_5000`)

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
