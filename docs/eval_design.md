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
   the left image; the right image is stored when the source has one (CODa, JRDB, MuSoHu, simulator renders),
   and simulator renders use the robot's own stereo baseline, field of view and camera height.
2. **(owner) Lidar and 3D ground truth are for building and scoring the set only.** Collisions are checked
   against lidar point clouds and boxes (real data) or scene meshes (simulator). A scenario without lidar or
   a mesh cannot enter the set.
3. **(owner) The task comes as a prompt.** Either a point goal, "please walk towards the goal (x, y)", or a
   natural-language instruction. Goal coordinates are in the robot frame at the current step: robot at
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
- **(owner) Always, with or without a reference:** smoothness (jerk, curvature, heading rate), efficiency
  (progress towards the goal per metre travelled; in closed loop, success within a radius, SPL and time to
  goal) and task completion.
- Target speed is about 0.5 m/s, slower than walking people (about 1.4 m/s). Reference paths are compared by
  distance along the path, not by time.

## Composition

- **(owner) Indoor is the majority: at least 100 scenarios. Outdoor at most 50**, kept to study the model, not
  for the robot's indoor use.
- **Size.** Detectable difference in collision rate between two models (15% base rate, 80% power, unpaired
  test): 100 scenarios: about 16 percentage points; 150: about 13; 200: about 11; 400: about 8. Paired
  comparisons and continuous scores (clearance, progress) detect smaller differences. **Proposal: start with
  100 indoor + up to 50 outdoor, and grow indoor to 150 if the confidence intervals of the first real model
  comparisons are wider than the differences that matter.** The rule is fixed now, before any result.
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

## Open questions for the owner

- Robot specification: stereo baseline, camera resolution, field of view, camera height, footprint radius
  (the evaluator assumes 0.3 m), whether it walks (legged / humanoid) or rolls.
- 100 or 150 indoor scenarios (proposal above).
- Language instructions in the first version, or point goals only?
- Most candidate sources are non-commercial (CC BY-NC-SA, HSSD CC BY-NC, Matterport academic only):
  confirm the startup's use fits.
