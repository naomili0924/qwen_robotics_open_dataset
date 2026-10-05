# Evaluation set: design and acceptance criteria

Written on 2026-10-04, before any model was trained, so that the selection cannot be tuned to results.
Requirements marked **(owner)** were set by the owner on 2026-10-04; the rest is the proposal built on them.
Open questions are at the end.

## Purpose

Rank training choices (data sources, with or without lidar supervision, supervised / PPO / GRPO) correctly
for an **indoor** robot, with a set small enough to audit every scenario by hand.

## Hard rules (never broken)

1. **(owner) The model sees cameras only.** Inputs: camera images, the robot's own past poses, and a text
   prompt. No lidar, depth or maps at inference. The robot carries a **stereo camera**: every scenario stores
   the left image; the right image is stored when the source has one (CODa, JRDB, MuSoHu, simulator renders).
2a. **(owner) The set is robot-agnostic.** The model is meant to be robust across robots, so camera height,
   field of view, stereo baseline and footprint are not fixed by the set: real scenarios keep their source's
   camera, simulator scenarios vary camera height and field of view. Scoring assumes a **humanoid walker like
   Tesla Optimus or Figure** (about 1.7 m tall, 0.3 m footprint radius, overhead clearance 2 m) that only
   walks (no running) and can **turn in place by any angle**. The footprint is a scoring parameter applied at
   evaluation time, never baked into the stored data, so other robots can be scored by changing it.
2. **(owner) Lidar and 3D ground truth are for building and scoring the set only.** Collisions are checked
   against lidar point clouds and boxes (real data) or scene meshes (simulator). A scenario without lidar or
   a mesh cannot enter the set.
3. **(owner) The task always comes as a text prompt, and the goal is part of the prompt.** Either a point
   goal, "please walk towards the goal (x, y)", or a natural-language instruction, which may also name a
   point ("walk to the door on your left at (3.0, 1.5)"). Language instructions are in the first version. Goal coordinates are in the robot frame at the current step: robot at
   (0, 0), x forward, y left, metres.
4. **The goal must not give away the answer.** Today's `goal` column is the end of the scored 5 s future, so
   "straight to goal" is near the recorded path by construction. In the evaluation set the goal is a point
   further along the reference path, beyond the scored horizon, except in scenarios whose subject is stopping
   at the goal.
5. **No overlap with training data.** Scenes, houses and recording sequences used in the evaluation set are
   removed from every training split. At most a few scenarios per sequence or house, so scenarios are close to
   independent.
6. **Every scenario is solvable and correctly labelled.** The reference path does not collide, and every
   scenario is inspected by hand (rendered images, path, obstacles) before it enters the set.

## Scoring

- **Collision** (hard failure): against moving boxes, stationary objects, lidar points and meshes; robot
  footprint radius from the robot's specification. Existing evaluator: `hnod/eval.py`.
  - Real data shows only what the lidar saw; a predicted path into unobserved space is reported
    (`unknown_fraction`) and counted as unverified, not as safe.
  - In replayed real data, people do not react to the robot. Only the simulator is closed loop.
- **(owner) Reference behaviour:** a human or expert path where the source has one (human-driven robot,
  person walking). Distance to it (ADE / FDE) is a secondary score: several paths can be good.
- **(owner) Always, with or without a reference:** smoothness (jerk, oscillation, needless stops; sharp or
  in-place turns are not penalised, because the robot can turn by any angle), efficiency
  (progress towards the goal per metre travelled; in closed loop, success within a radius, SPL and time to
  goal) and task completion.
- Target speed is about 0.5 m/s, slower than walking people (about 1.4 m/s). Reference paths are compared by
  distance along the path, not by time.

## Composition

- **(owner) Indoor is the majority: at least 100 scenarios. Outdoor at most 50**, kept to study the model, not
  for the robot's indoor use.
- **Size.** Detectable difference in collision rate between two models (15% base rate, 80% power, unpaired
  test): 100 scenarios: about 16 percentage points; 150: about 13; 200: about 11; 400: about 8. Paired
  comparisons and continuous scores (clearance, progress) detect smaller differences. **(owner) Start with
  100 indoor + up to 50 outdoor.** Grow indoor to 150 if the confidence intervals of the first real model
  comparisons are wider than the differences that matter; the rule is fixed now, before any result.
- **Coverage** (each scenario tagged; selection fills every tag before adding more of any one):
  - indoor layout: corridor, doorway or narrow passage, open room or lobby, cluttered furniture, corner or
    turn, goal behind the robot, goal out of sight (around a corner or in another room), stairs or
    drop-offs nearby, glass or mirrors, dim or bright lighting;
  - people: none, static people, person crossing, oncoming, overtaking, crowd;
  - task: point goal, language instruction, stopping at the goal;
  - outdoor: walkway, crossing, crowd, ramp or kerb.
- **Candidate sources.**
  - Indoor, real, with lidar: CODa and JRDB indoor sequences (published), MuSoHu (person-worn ZED 2 stereo
    + Velodyne, CC0), SCAND (cameras + Velodyne; collisions from raw lidar scans, since it has no tracks).
  - Indoor, simulated: HSSD (published point-goal set, exact meshes, closed loop), HM3D (real scans,
    gated, academic use only).
  - Outdoor: CODa, JRDB and RoboSense outdoor sequences, SCAND, MuSoHu.
- **Selection.** Among valid candidates, prefer scenarios where naive baselines (stationary, constant velocity,
  straight to goal, a model that sees past poses but no images) fail and the reference succeeds. Check the
  final set on models that took no part in the selection, so the set measures navigation and not only "is not
  constant velocity".

## Acceptance tests (the set is frozen only when all pass)

1. A model that sees no images (past poses + goal only) scores clearly worse than image models.
2. Models whose order is known (stationary < constant velocity < straight to goal < reference; checkpoints
   trained on 10 / 30 / 100% of the data) come out in that order with separated confidence intervals.
3. Ranking on the set agrees with ranking on all held-out candidates (Kendall tau), after weighting.
4. Ranking on the simulator part in open loop agrees with closed-loop ranking in the simulator.
5. Re-running the selection with another seed changes no conclusion.

## Settled with the owner (2026-10-04)

- Robot: a walking humanoid (Optimus / Figure class), turns in place; the set itself stays robot-agnostic.
- Size: start with 100 indoor + at most 50 outdoor.
- Prompts: every scenario has one; language instructions included from the first version.
- Licences: not a constraint during the current broad investigation; recorded per source for later.

## Open

- Output format: an (x, y) path cannot show an in-place turn. Whether predictions should carry heading
  (x, y, yaw) matters for closed loop; it does not change open-loop collision scoring.

## As built: v1 (2026-10-05)

Code: `scripts/build_eval_suite.py` (candidates, select, write), `hnod/suite.py` (protocol and scoring),
`scripts/evaluate_suite.py` (score predictions), `vla/predict_suite.py` (score a trained policy). Published as
the Hugging Face dataset `Jinyan0924/qwen_robotics_nav_eval`, config `v1`.

- **Candidate pool, all held out from training:** CODa `coda_2hz` test + validation, JRDB `jrdb_2.5hz` all
  splits, RoboSense `robosense_1hz` validation (it has no labelled test split), HSSD `hssd_2hz` test +
  validation houses. JRDB as a whole is reserved for evaluation: it adds only about 1,300 frames to training
  but holds most real indoor scenes with people. Training uses the train splits of CODa, RoboSense and HSSD
  only (their per-frame configs publish only `train`).
- **Protocol.** The model returns a path; the robot follows it at 0.5 m/s. The goal in the prompt lies on the
  recorded reference path 3–10 m beyond the end of the scored horizon (or at the episode's end for simulated
  episodes), so the goal no longer gives away the answer.
- **At-fault collisions.** Recorded people do not react to the robot. Following a recording robot that moved at
  about 1 m/s with a 0.5 m/s robot, people who walked behind it run into the slower robot (in one JRDB sequence
  68% of scenarios had someone walk through a robot that stood still). As in nuPlan, a contact with a moving
  agent counts only if, at first contact, the robot is moving and the agent is ahead of it. Contacts with
  standing people, objects, the static map and lidar points always count.
- **Environment labels.** CLIP alone was not reliable: it called the JRDB Clark Center courtyard and Memorial
  Court, and every RoboSense covered walkway, "indoor". Every real candidate that CLIP did not call clearly
  outdoor (287) was reviewed by eye; the labels are in `docs/eval_audit_v1.json`. HSSD point-goal episodes
  partly run through the gardens around the houses (the generator's room-polygon restriction leaks); simulated
  scenarios therefore need CLIP indoor probability >= 0.95, and near-duplicate views of one house (episodes
  starting from the same pose) are removed by CLIP image-embedding similarity (> 0.94).
- **Composition.** 100 indoor + 50 outdoor. Real indoor data is the binding constraint: after the 5 s spacing
  rule and the checks below, the held-out real recordings yield only 25 independent indoor scenarios (CODa 8,
  JRDB 17), so 75 indoor scenarios are simulated (HSSD houses, no people). Outdoor: RoboSense 33, CODa 10,
  JRDB 7. Real scenarios are selected first; then tag coverage, then scenarios where naive baselines fail.
- **Solvable as scored.** A scenario is valid only if its reference path, followed at 0.5 m/s under the same
  at-fault rules, is collision-free and does not reverse, and the goal is at least 2 m away. Map collisions get a
  5 cm tolerance (half a map cell): the HSSD planner grazes walls at exactly the robot radius, and 1–2 cm of map
  quantisation decided whether its own path "collided". With this, the reference never collides; it succeeds in
  79% (it is not the most direct route to the prompt's goal), naive planners in 20–23% indoors.
- **Audit.** Every scenario was inspected on an audit image. Rejections (12 scenarios, 2 recordings whose
  operator reverses or loops) and the 150 accepted scenarios are in `docs/eval_audit_v1.json`; the selection keeps
  accepted scenarios fixed, so re-running it reproduces the published set and a change needs only the
  replacements reviewed.
- **Biggest limitation and the v2 plan.** Indoor results are dominated by simulated houses without people; the
  25 real indoor scenarios (with crowds) are reported separately but are indicative only. v2 should add real
  indoor recordings with lidar (MuSoHu, SCAND indoor parts) and HM3D (real scans) to replace simulated ones.

## v2 (2026-10-05)

All 100 indoor scenarios real. MuSoHu indoor walks (helmet-worn VLP-16 and ZED 2 at about 1.7 m, CC0) were converted
with `scripts/convert_musohu.py`: sweeps levelled by the floor plane each sweep sees (the odometry's tilt left up to
5 degrees, which turned distant floor into obstacles), people tracked in the lidar (`hnod/lidar_tracks.py`). 494
candidates were labelled indoor / outdoor by eye; Trader Joe's (floor estimated at 1.0 m, shelves) and one recording
without a detected floor were rejected. Selection kept v1's 75 accepted real scenarios and filled the 75 simulated
slots with MuSoHu (at most 15 per recording); all 75 passed the audit.

Consequence: naive baselines succeed far more often (straight to goal 69% indoor vs 20% in v1). Real indoor walks
are mostly straight corridors and halls; harder real scenes (doorways, turns, dense crowds) are the gap for v3.
SCAND could not be fetched from this machine (403 from its Dataverse); GND is outdoor-only.
