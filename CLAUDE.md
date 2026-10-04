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

## Source status

- CODa, RoboSense, JRDB: converted and published. JRDB came from the Stanford mirror with the owner's
  explicit go-ahead; its wheel odometry is refined by ICP and its lidar offsets were re-derived.
- RoboSense has no test split (source test labels are withheld); the owner declined re-splitting.
- SiT: blocked. The download needs the owner to sign the authors' terms form, and the licence statements
  conflict (ND vs SA); confirm with the authors before publishing converted data.
- SCAND, MuSoHu: no object tracks, so not usable as collision ground truth.
- Habitat + HSSD point-goal: generated with `scripts/generate_habitat_pointnav.py`. HSSD stood in for HM3D
  because HM3D needs a Matterport API token.
- HM3D, MP3D (VLN-CE, object-goal), nuScenes: wait for the owner's credentials (see README).

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
