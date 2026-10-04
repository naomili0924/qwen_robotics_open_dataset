# Qwen-RobotNav and Qwen-VLA: data formats

Read on 2026-10-04 by a research pass through a summarising fetch tool, **not checked against the PDFs**.
Re-read the papers before hard-coding any number. Neither model has released weights, code or data.

Sources: Qwen-RobotNav arXiv 2606.18112 (v3), github.com/QwenLM/Qwen-RobotNav ("no plan to release the
model weights"); Qwen-VLA arXiv 2605.30280 (v2), github.com/QwenLM/Qwen-VLA.

## Qwen-RobotNav

- **Inputs.** N cameras x T timesteps; single front camera or front/left/right/rear. History grows online;
  T' <= T frames sampled in "random / latest" modes. Visual token budget U[2048, 4096], temporal decay
  weights, per-camera weights (front 2.0, right 1.0, back 0.5, left 1.0); frames rescaled to their pixel
  budget. Prompt: embodiment preamble ("Imagine you are a robot programmed for navigation tasks"), task mode
  (VLN, PointNav, ObjNav, Tracking), "Time step t Front View <image>" tags, instruction. ObjNav template
  "navigate to the {goal_object}" / "find and reach the {goal_object}". PointNav: goal as numeric egocentric
  coordinate plus current pose, distance, bearing. Not stated: resolution, max frames, frame rate.
- **Outputs.** 8 waypoints x (x, y, theta) = 24 numbers; 4-layer MLP (hidden 512, GELU) on the final hidden
  state; normalised to [-1, 1] by the per-dataset 99th percentile of each coordinate; MSE + lambda * VL loss
  (lambda 1.0). "Stop actions are always retained." Not stated: waypoint spacing, theta convention, how stop
  is encoded.
- **Training data (15.6 M samples).** VLN-CE R2R 1,491K; VLN-CE RxR 4,140K; point-goal (Habitat, MP3D + HM3D)
  984K; object-goal (MP3D + HM3D-OVON) 2,000K; tracking (EVT-Bench) 1,486K; driving (nuScenes + OpenScene)
  3,216K; in-house text-to-video 40K; vision-language about 2.34 M. Augmentation: image style transfer,
  instruction paraphrasing, random camera height / FOV / aspect ratio.
- **Training.** From Qwen3-VL, 2B to 8B, end to end, supervised only. AdamW, lr 2e-5 backbone / 1e-4 head,
  batch 256; 2,816 H100 GPU hours for the 8B model.
- **Evaluation (closed loop).** VLN-CE R2R / RxR (NE, OS, SR, SPL), VLNVerse, VLN-PE, ObjNav MP3D / HM3D /
  HM3D-OVON, EVT-Bench (TR, CR, SR), NAVSIM (PDMS 91.4), embodied QA, zero-shot on a Unitree Go2.

## Qwen-VLA

- **Inputs.** One or more frames / history windows, views tagged by source. Prompt: "The robot is
  {robot_tag} with {arms}[, waist][, and mobile base]. The control frequency is {FPS} Hz. Please predict the
  next {chunk_size} control actions to execute the following task: {instruction}."
- **Outputs.** Navigation: 8 waypoints x (dx, dy, dtheta), "relative displacement and heading change in the
  ground plane". Per-dataset quantile normalisation (1st / 99th percentile, clipped to [-1, 1]). DiT-style
  flow-matching head (about 1.15 B parameters, 16 blocks), a few Euler steps. Not stated: whether deltas are
  consecutive or relative to the current pose, spacing, stop handling.
- **Training data (shares of the mix).** Manipulation 74.2%; navigation 7.5% (instruction following 4.3%,
  object search 2.3%, tracking 1.0%); human egocentric 6.0%; synthetic 3.7%; VL 3.4%; grounding 2.5%;
  driving 2.4% as question answering only.
- **Training.** Qwen3.5-4B backbone; four stages ending in RL: PPO with GAE (eps 0.2, gamma 0.99), sparse
  binary success rewards in SimplerEnv (manipulation, not navigation).
- **Evaluation.** Closed loop; navigation R2R SR 53.8 / 57.5, RxR SR 55.1 / 59.6 (Base / Instruct).

## What a stored sample needs to serve both

- Front camera history (side / rear optional), at native resolution, with timestamps; as long as the source
  has (both models resample it).
- Raw metric robot pose (x, y, yaw) for every past and future step; **not** normalised, **not** resampled.
  RobotNav's (x, y, theta) and Qwen-VLA's (dx, dy, dtheta) are derived at load time, as are both
  normalisations (99th percentile; 1st / 99th quantiles), computed per dataset.
- Task type, instruction, goal (object category, target description, or egocentric coordinate), stop /
  terminal flag, embodiment name, control rate (Qwen-VLA's prompt needs the FPS).

Label for single-model data (`fits`): `robotnav` only = point-goal with numeric coordinates and driving
trajectories (Qwen-VLA has no point-goal task and uses driving only as QA); `both` = instruction following,
object search, tracking; `vla` only = manipulation (out of scope).

## Consequences for our design (decided 2026-10-04)

- The target robot walks at about 0.5 m/s. Waypoints and history are taken **by distance along the path**
  at load time, so a source's speed is not baked in. Default under discussion: 8 waypoints 0.25 m apart.
- Store data per frame (one row per frame: image, pose, timestamp) plus a small table of segments
  (episode, frame range, task, instruction); windows are cut at load time. The 21-step "scenario" rows
  stay for the lidar-verified evaluation sets.
