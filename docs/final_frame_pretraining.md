# Final-frame pretraining

Direction set by the owner on 2026-10-05, after the first run (`e1_all_sqrt`) showed that a goal cut from the
recorded future, given as text or coordinates, is not read by the model
(evidence: https://claude.ai/artifact/BpCSTitQ9x8uijCMfmV93w, Chinese: https://claude.ai/artifact/HLP2CMw1cqx1TjfyozmZAT).

## Task

A pretraining stage that teaches how what the camera sees relates to how the camera moves. It is not the
onboard model: at deployment nobody has the future frame.

- **Input:** 5 past frames 1 s apart, the current frame, and **one final frame 5 s ahead** (7 images), plus an
  embodiment prompt in the style of Qwen-VLA (arXiv 2605.30280, section 2.3): "The camera is carried by a human
  walking. The past views are sampled at 1 Hz. Please predict the next 10 positions at 2 Hz." (or "by a robot").
  No task or instruction, nothing numeric: no past positions, no speed, no goal coordinates.
- **Output:** 10 positions (x forward, y left, metres, relative to now), one every 0.5 s: the recorded motion,
  timing included, so slowing for people is part of the target.
- **Embodiment:** EgoWalk = human; CODa, RoboSense, HSSD = robot. In the evaluation set MuSoHu = human.
- **Samples are cut at load time** (`hnod/windows.py`, `mode="final_frame"`); nothing new is stored. Every frame
  followed by 5 s of recording is a sample. Point-goal and language prompts are not used in this stage.
- **Model:** `Qwen/Qwen3.5-4B-Base` (the Qwen-VLA backbone family, Apache-2.0), language model fully
  fine-tuned, vision encoder frozen, action head the existing MLP regression head. On one 94 GB H100 that needs
  gradient checkpointing and 8-bit AdamW (`--optimizer adamw8bit`): about 44 GB and 1.8 samples/s with the
  `flash-linear-attention` kernels installed (`causal_conv1d` has no wheel for CUDA 13 here; with it the model
  would be faster). Qwen3-VL-2B-Instruct runs at 5.9 samples/s and was used for the first pipeline test.

## Commands

```bash
bash scripts/setup_machine.sh                         # new machine; needs HF_TOKEN in /workspace/.env
bash scripts/run_ff.sh <run name> <steps>             # train; resumes from the Hub copy if there is one
python -m vla.eval_final_frame --hub-repo Jinyan0924/qwen_robotics_nav_policy \
    --checkpoint <run name>/last --out eval.json      # evaluation with the input-use checks
python scripts/build_final_frame_eval.py --version v2 --upload    # rebuild the evaluation config
```

Checkpoints: `Jinyan0924/qwen_robotics_nav_policy`, one folder per run (`<run>/last`, weights in bf16; the
optimizer state of a full fine-tune is not uploaded, so a resumed run restarts Adam). `e1_all_sqrt/` is the
earlier LoRA run with goals in the prompt, stopped at step 12,000 and kept as a baseline.

## Evaluation

`v2_final_frame` config of `Jinyan0924/qwen_robotics_nav_eval`: 134 of the 150 suite-v2 scenarios (93 indoor, 41
outdoor) that have a camera frame at the end of the horizon. Scored at the recorded timing (`hnod/suite.py`,
`score_timed`): collision (at-fault rule for moving people), completion (end within 1 m of where the recording
ended), smoothness (heading oscillation, peak acceleration), and distance to the recorded trajectory.

Every evaluation also runs three input-use checks: the final frame swapped for another scenario's, the final
frame replaced by the current frame, and the human / robot sentence swapped.

Baselines: recorded trajectory 100% success; constant velocity 62% (it is given the true velocity, which the
model is not); standing still 1%.

## Known limits

- The final frame comes from the path that was actually taken; a scene still has one outcome.
- Human data is EgoWalk only and robot data is CODa, RoboSense and simulated HSSD, so "human or robot" is
  entangled with place and camera.
- The per-frame HSSD data lacks the last 5 s of each episode.
- JRDB scenarios have a 4 s horizon; the model is trained on 5 s.

## End-to-end test (2026-10-06, `ff0_e2e_test`)

Qwen3-VL-2B, 500 steps of 16 samples (8,000 samples, under 1% of the data), one H100: 2.9 s per step, 82 GB.
Training loss 0.10 -> 0.02, validation 0.029 -> 0.026. Checkpoint (3.4 GB) uploaded, reloaded from the Hub, evaluated:

| | Success | Completed | Collided | ADE | End error | Speed |
|---|---|---|---|---|---|---|
| Recorded trajectory | 100% | 100% | 0% | 0 | 0 | 1.00 m/s |
| Constant velocity (given true velocity) | 62% | 70% | 20% | 0.49 m | 1.12 m | 0.99 |
| Model | 29% | 33% | 19% | 0.88 m | 1.71 m | 0.87 |
| Model, final frame swapped | 29% | 33% | 19% | 0.89 m | 1.71 m | 0.87 |
| Model, no final frame | 28% | 32% | 20% | 0.88 m | 1.71 m | 0.87 |
| Model, human / robot sentence swapped | 26% | 29% | 16% | 0.93 m | 1.77 m | 0.86 |

The pipeline works end to end. The model does not use the final frame yet: changing it moves the predicted end
point by 0.05-0.07 m (the sentence: 0.41 m). 8,000 samples is far too few to judge, but this check is the one to
watch in any longer run.
