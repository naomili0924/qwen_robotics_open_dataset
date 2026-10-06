# Project notes for agents

Context that is not derivable from the code. Keep it short and current; delete what stops being true.

## What the owner wants

- **Scenario layout (owner's correction, 2026-10-02).** The model is image-in, trajectory-out. Past =
  images only (10 past + current, front camera, full resolution) plus the robot's own past poses.
  Future = 3D ground truth like the Waymo Open Dataset: boxes with a static/dynamic flag and per-step
  lidar points labelled ground / static / dynamic, for the current + 10 future steps. Every source uses
  this schema (`hnod/io.py`), one Hugging Face dataset repo per source.
- **Policy library (`vla/`).** A Hugging Face-pipeline-like front end for training and inference with a
  switchable action head (MLP regression vs flow with MLP or DiT denoiser), attachable task heads to test
  how transferable the embedding is, per-layer probing (`vla.probe`), PPO / GRPO (`vla.rl`). Do not tune
  results or spend GPU hours on long runs unless asked; fix bugs.
- **Workflow.** Every change goes on a branch with a pull request; never push to `main` directly.
- **Scale target.** Reproduce Qwen-RobotNav's training recipe (about 15.6 M samples) from public sources:
  VLN-CE R2R/RxR, HM3D/MP3D object-goal, EVT-Bench tracking, nuScenes/OpenScene driving, point-goal.
  At that size data should be generated on the fly or streamed, not stored as one upload.

## Direction set by the owner on 2026-10-04

Purpose: an end-to-end navigation policy from open weights and open data (images in, trajectory out,
perfect control assumed), whose results tell a robotics startup how to collect training data (is lidar
worth it?) and how to build an evaluation set. It does not have to succeed; it has to be informative.

- **Training data: as much as possible, from every public navigation-relevant dataset**, even ones that may
  turn out to be poor; whether a source helps is itself a result. Trajectory-only sources (camera + ego
  poses, no lidar) count as training data.
- **Evaluation data (owner, revised 2026-10-04): start with about 100 scenarios, indoor the majority (at least
  100; grow if that is too few to tell models apart), outdoor at most 50.** Replaces the earlier 400. Cameras
  only at inference (robot has a stereo camera, no lidar); lidar / meshes only for building and scoring the
  set. Task given as a prompt, goal in the robot frame with the robot at (0, 0). Human / expert reference
  when available; always score collision, smoothness, efficiency. Robot-agnostic set; scoring assumes a humanoid walker
  (Optimus / Figure class, turns in place); every scenario has a text prompt containing the goal; licences
  are not a constraint during the current broad investigation. Full design, acceptance tests and open
  questions: `docs/eval_design.md`.
- **Target robot speed: about 0.5 m/s, below 1.0 m/s** (owner, 2026-10-04; an earlier "1.5 m/s" was
  withdrawn). Slower than every walking-person dataset (about 1.4 m/s) and most wheeled-robot sets. Keep
  speed out of the stored targets: store raw metric poses with timestamps and each sample's source speed,
  and derive waypoints at load time by distance along the path or at the target speed. Simulator runs
  should use about 0.5 m/s (the first 113 HSSD houses used 1.0 m/s). Closest real matches by speed: slow
  indoor robots (GoStanford, HuRoN, JRDB).
- **One preprocessed format usable by both Qwen-RobotNav and Qwen-VLA.** Samples that can only serve one
  of the two are still created and labelled as such.
- **Everything is published on Hugging Face, ready to load.** Check each source's terms first; some forbid
  redistribution. The owner accepted (2026-10-04) **gated repos for anything rendered in Matterport scenes**
  (MP3D, HM3D and episodes built on them): the access form must show the Matterport academic EULA and
  record who accepts. Access/licence survey of about 70 sources:
  https://claude.ai/artifact/WGnLK4P5b8Wa14cCdcTAwT (not publishable: Waymo, ONCE, CoVLA, NVIDIA AV,
  Ego4D, Aria AEA/ADT, SiT, ScanNet; YouTube-based sets: ids and derived paths only).
- Model architecture stays fixed (one pretrained VLM); the variables are data and training method
  (supervised, PPO, GRPO).
- Open evaluation problems I raised and the owner called "good": the goal column leaks the answer (it is the
  end of the 5 s future), real test sets are too easy/small, collision rate alone rewards standing still,
  nothing is closed loop yet. Review page: https://claude.ai/artifact/ECWC19658KhJfe6znx3Zt5

## Machines are disposable

The owner rents GPUs and switches machines. Assume everything outside GitHub and the Hugging Face Hub is
lost: `/dev/shm` (where large builds run) is a RAM disk, and `/workspace` is usually not a volume.

- Code: GitHub `naomili0924/qwen_robotics_open_dataset`.
- Data: Hugging Face user `Jinyan0924`: `qwen_robotics_open_dataset` (CODa), `..._robosense`, `..._jrdb`,
  `habitat_hssd_pointgoal_nav_scenarios`. Generators upload finished shards while running
  (`scripts/sync_parts_hf.py`).
- Checkpoints: train with `--hub_repo <repo> --resume auto` (`vla/hub.py`).
- Credentials: `${WORKSPACE}/.env` holds `HF_TOKEN`, `GITHUB_TOKEN`, optionally `MATTERPORT_TOKEN_ID` /
  `MATTERPORT_TOKEN_SECRET`. The owner re-enters them on a new machine. Never print them.
- New machine: `bash scripts/setup_machine.sh [--habitat]`.

## Direction set by the owner on 2026-10-05 (evening): final-frame pretraining

Read `docs/final_frame_pretraining.md` first. In short: run `e1_all_sqrt` showed the model does not read a goal
cut from the recorded future (text or coordinates), so the owner replaced that recipe with a pretraining stage:
past frames + **one frame 5 s ahead** -> the recorded positions in between (timed), prompt says only "human" or
"robot", **no numeric inputs**, language model fully fine-tuned, vision frozen. It is for learning how views and
motion relate, not for onboard use. The owner's words: start with a small model and a subset for fast iteration;
no dataset annotation or re-processing at this stage; old checkpoints stay, new runs get new Hub folders; test
end to end on a sample first, then discuss scale (more GPUs may be rented).

## State on 2026-10-06

`main` holds everything through PR #5. Branch `final-frame-pretraining` (PR open) holds this stage.

- **Stopped:** `e1_all_sqrt` (LoRA, goal in prompt + numbers) at step 12,050; checkpoints kept on the public model
  repo `Jinyan0924/qwen_robotics_nav_policy` (`e1_all_sqrt/last` = step 12,000, `step_5000`, `step_10000`, `best`).
- **Built:** final-frame samples at load time (`hnod/windows.py` `mode="final_frame"`), items with images +
  embodiment sentence only (`vla/data.py`), full fine-tune checkpoints as bf16 safetensors without optimizer state
  (`vla/model.py`, `vla/hub.py`), evaluation config `v2_final_frame` (134 scenarios) with timed scoring
  (`hnod/suite.py score_timed`) and input-use checks (`vla/eval_final_frame.py`), `scripts/run_ff.sh`.
- **End-to-end test `ff0_e2e_test`** (Qwen3-VL-2B, 500 steps x 16 samples, 2.9 s/step, 82 GB on one H100):
  train loss 0.10 -> 0.02; checkpoint reloaded from the Hub and evaluated. Success 29% (constant velocity 62%,
  recorded 100%). **Input-use checks: swapping or removing the final frame moves the predicted end point by only
  0.05-0.07 m; swapping the human / robot sentence moves it 0.41 m.** After 8,000 samples the model does not yet
  use the final frame. Numbers: `/workspace/runs/eval/ff0_step500.json` (also on the model repo under `ff0_e2e_test/eval/`).
- **Backbone decision (owner, 2026-10-06):** start with Qwen3.5-2B-Base, move to Qwen3.5-4B-Base (the Qwen-VLA
  backbone) once the 2B shows it learns; the owner may rent a bigger GPU for the 4B. `scripts/run_ff.sh <run>
  <steps> 2b|4b`. Prompt is Qwen-VLA style: who carries the camera + frame rates, no task.
- **`ff2_qwen35_2b`** (2B full FT on all 1.11 M frames, 4.1 s/step): stopped by the owner at step 3,450 (GPU wanted for
  the next run); milestones 1,000-2,500 + `best` (step 3,000) on the Hub. It never used the final frame (swapping it
  moved the prediction 0.03 m); success 16-26%.
- **Dedup set + descriptions (2026-10-06):** `Jinyan0924/qwen_robotics_nav_pretrain_dedup` = 49,391 samples chosen
  motion-first (`scripts/build_dedup_index.py`, `select_dedup.py`, `publish_dedup.py`), with a `samples` table
  (prompt, system, motion class) and an `annotations` table: Claude Haiku 4.5 descriptions of each sample from 11
  frames (-5 s..+5 s) + the recorded path (`scripts/annotate_samples.py`, Batch API, $89 for all). Descriptions are
  a training *target* (`--text-loss W`: assistant text after the prompt, CE on its tokens; the trajectory head still
  reads the last prompt token), never an input. `ANTHROPIC_API_KEY` lives in `/workspace/.env`.
- **Stopped by the owner at step 4,325 (19:45 UTC): `ff3_lora4b_dedup`** (started 09:04 UTC 2026-10-06): Qwen3.5-4B-Base, LoRA rank 32 on attention +
  linear-attention + MLP (`run_ff.sh ... 4b-lora`), dedup set, `--text-loss 0.05`, 6,200 steps (2 epochs), 8.4 s/step,
  19 GB. **It uses the final frame:** at step 3,000 (1 epoch) success 0.59 / completed 0.65 / collided 0.19 / ADE 0.59 /
  FDE 0.99 (constant velocity 0.62 / 0.70 / 0.20 / 0.49 / 1.12); swapping the final frame drops success to 0.10 and
  moves the end point 2.4 m; swapping the prompt 1.2 m. Milestones 1,000-5,000 with evals under
  `ff3_lora4b_dedup/` on the model repo; `step_4000` = `best` = `last`. Training a 4B here is slowed by the missing
  `causal_conv1d` wheel (CUDA 13). How to resume, start from its weights (`--init-from hub:...`) or evaluate:
  `docs/final_frame_pretraining.md`, "Reusing the final-frame checkpoint".
- **Camera calibration in the prompt (owner, 2026-10-06):** `--camera-prompt` adds the field of view (from K) and the
  height above the ground to the embodiment sentence, **from stored calibration only**; an estimated or missing value is
  written as "unknown" (MuSoHu's helmet height was estimated from the lidar and is therefore unknown to the model).
  Tables carry `camera.height_source` (episodes), `camera_prompt` (dedup samples, final-frame eval configs). Not yet used
  by a trained run.
- **Running: `ff4_lora4b_instruction`** (started 19:50 UTC 2026-10-06, owner's request): same base, LoRA and data as
  ff3, **no final frame**; the annotation (description, place, interaction) is the task text in the prompt
  (`--no-final-image --instruction description,place,interaction`). Extended to 6,200 steps (two epochs, restarted from
  step 1,000 with the longer schedule), milestones every 1,000 scored on v2 and v3 with their annotations. At step
  1,000: v3 success 0.41 / completed 0.56 / ADE 0.67 / FDE 1.03; swapped text -> 0.07 (reads the text), collisions
  unchanged (0.26). **Queued after it: `ff5_lora4b_instruction_camera`** = ff4 + `--camera-prompt` (ablation, auto-start
  via `/workspace/logs/launch_ff5.sh`). Exact commands and how to resume:
  `docs/final_frame_pretraining.md`, "Instruction-conditioned variant". Caveat: the text states the outcome, so the
  score measures text following; a motion-free rewrite of the descriptions would be the stricter test.
- **Eval v3 (2026-10-06):** `v3_final_frame` = 132 motion-diverse real scenarios (`scripts/build_eval_v3.py`,
  `docs/eval_audit_v3.json`); v2 was two thirds straight. Constant velocity on v3: success 0.33, ADE 0.84, FDE 1.92.
  ff3 at step 3,500 on v3: success 0.41, completed 0.42, collided 0.24, ADE 0.75, FDE 1.32 - the first run that beats
  constant velocity; swapped final frame moves the end point 2.6 m. Eval scenarios (v2, v3) carry Claude descriptions
  (`v2_annotations`, `v3_annotations`) and `future_images` at 1 Hz. The watcher scores milestones on v2 and v3.
- **Hardware notes:** this machine = 1x H100 NVL 94 GB, 128 CPUs, 377 GB RAM, 32 GB root disk (always short:
  base-model weights go to `HF_HUB_CACHE=/dev/shm/hf_hub`, runs to `/dev/shm/runs`). Throughput: 2B 4.2
  samples/s (5.2 without checkpointing at 90 GB), 4B 1.8 samples/s (needs 8-bit AdamW; `causal_conv1d` has no
  wheel for CUDA 13 here, so Qwen3.5 is slower than it could be). One pass over all 1.11 M frames: 2B about
  3 days, 4B about 7 days on this GPU. **Multi-GPU training is not implemented** (single process; would need
  DDP + rank-sharded streaming in `vla/stream.py`).
- **Open with the owner:** whether the gain of ff3 over ff2 comes from the dedup set, the description target or LoRA
  (one ablation each would tell); how to beat constant velocity on ADE (more epochs, the 1.06 M remaining frames with
  descriptions, a bigger GPU for the 4B without checkpointing); multi-GPU training is not implemented.
- **Machine notes:** root disk is 32 GB and nearly full (model caches); run directories and downloads go to
  `/dev/shm`. The flash-attention kernel cannot load from `/dev/shm` (noexec), so `HF_HOME` stays on the root disk.

## Source status

- CODa, RoboSense, JRDB: converted and published. JRDB came from the Stanford mirror with the owner's
  explicit go-ahead; its wheel odometry is refined by ICP and its lidar offsets were re-derived.
- RoboSense has no test split (source test labels are withheld); the owner declined re-splitting.
- SiT: blocked. The download needs the owner to sign the authors' terms form, and the licence statements
  conflict (ND vs SA); confirm with the authors before publishing converted data.
- SCAND, MuSoHu: no object tracks, so not usable as collision ground truth.
- Habitat + HSSD point-goal: 113 of 168 houses generated with `scripts/generate_habitat_pointnav.py` and
  published (see the state section). HM3D is now accessible too (token works).
- HM3D, MP3D (VLN-CE, object-goal), nuScenes: wait for the owner's credentials (instructions were given
  on 2026-10-04: Matterport API token in `.env`, signed MP3D form -> `download_mp.py`, nuScenes login).
  The code exists (`--task vln|objectnav` in the generator, `hnod/nuscenes.py`) but has only been run on
  stand-ins (HSSD scenes, a synthetic nuScenes fixture); expect to debug on first contact with real data.
- EVT-Bench tracking and OpenScene: not started. Rows have `task` / `instruction` columns from the
  `episode-sources` branch on; datasets published before that lack them and loaders tolerate it.

## Things that bit before

- habitat-sim can die in native code; the generator runs one subprocess per scene and resumes from `.done`
  markers. More than about 64 CPU-rendering workers slow each other down badly.
- On 2026-10-03 the GPU of the rented machine went into "requires reset" while several EGL contexts were
  created at once; context creation is now serialised by a file lock. `--cpu` renders with Mesa llvmpipe
  through `tools/egl_software_only.c`.
- A recomputed navmesh also covers roofs and gardens; episodes are restricted to HSSD's room polygons.
  HSSD's closed door objects split houses into rooms (`scripts/prepare_hssd.py` removes them).
- Per-file downloads of large Hugging Face repos get rate limited (429); clone with git-lfs instead.
- `pkill -f <pattern>` matches the calling shell's own command line; kill by pid.
