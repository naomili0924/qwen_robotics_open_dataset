---
license: other
license_name: mixed-non-commercial
license_link: LICENSE.md
task_categories:
- robotics
tags:
- navigation
- evaluation
- benchmark
- indoor
- humanoid
pretty_name: Robot navigation evaluation suite
size_categories:
- n<1K
configs:
- config_name: v1
  data_files:
  - split: test
    path: data/v1/test-*.parquet
---

# Robot navigation evaluation suite (v1)

150 hand-audited scenarios for **image-in, path-out navigation policies of an indoor robot** (a walking
humanoid, about 0.5 m/s): 100 indoor and 50 outdoor, drawn from four open datasets, each with a text prompt
that contains the goal and 3D ground truth (lidar, tracked people, obstacle maps) for scoring. Built by
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset)
(`scripts/build_eval_suite.py`; design and decisions in `docs/eval_design.md`, every audit decision in
`docs/eval_audit_v1.json`).

![examples](figures/eval_examples.png)

*Each example: left, the current camera image with the recorded path; right, the bird's-eye view with the
static obstacle map (dark), people (red) with their future tracks, vehicles (purple), the recorded reference
path (green), the goal of the prompt (star) and the naive baselines (dashed).*

## Protocol

- **Input the model may use:** `past_images` (current + 10 past front-camera frames, oldest first), the robot's
  own past poses (`ego` x / y / heading at the past steps), the camera intrinsics, and `prompt`, e.g. "Please
  walk towards the goal (7.9, 2.6)." Coordinates: metres in the robot frame at the current moment, robot at
  (0, 0), x forward, y left. **Never** the lidar, maps, tracks or the future part of `ego`: those are ground truth.
- **Output:** a path, a list of (x, y) points in the same frame, any length and spacing.
- **Execution:** the robot follows the path at **0.5 m/s** with perfect control over the scenario's horizon
  (5 s for CODa and HSSD, 4 s for JRDB, 10 s for RoboSense); a path that ends early means it stops there.
- **Scores** (`hnod/suite.py`):
  - `collided`: hits a standing person or object, the static map (with a 5 cm tolerance for the 10 cm map grid),
    or, **at fault**, a moving person or vehicle: the robot is moving and the agent is ahead of it at first contact.
    Recorded people do not react to the robot, so being walked into is not counted (as in nuPlan).
  - `progress_ratio`: distance gained towards the goal over the most a perfect path could gain (1 = straight
    at full speed); `path_efficiency`: distance gained per metre walked.
  - `success`: no collision and `progress_ratio` >= 0.5.
  - `wiggle_rad`: heading oscillation (total minus net turning); turning in place is not penalised.
  - `reference_deviation_m`: distance to the recorded reference path, compared by distance along the path. Secondary:
    several paths can be good.
  - `collided_lidar` (raw lidar points) and `unknown_fraction` (share of the path over unobserved ground) are
    reported separately.

```bash
python scripts/evaluate_suite.py --pred my_predictions.json     # {"v1-0000": [[x, y], ...], ...}
python scripts/evaluate_suite.py --baseline straight_to_goal
python -m vla.predict_suite --checkpoint runs/<run>/last --out preds.json   # a policy trained with vla/
```

## Scenario distribution

![distribution](figures/eval_distribution.png)

| Environment | Source | Scenarios |
|---|---|---|
| indoor | coda | 8 |
| indoor | hssd | 75 |
| indoor | jrdb | 17 |
| outdoor | coda | 10 |
| outdoor | jrdb | 7 |
| outdoor | robosense | 33 |

**Environment**

| | Scenarios | Share |
|---|---|---|
| indoor | 100 | 67% |
| outdoor | 50 | 33% |

**People around the robot** (from the recorded tracks; a scenario can have several interaction tags)

| | Scenarios | Share |
|---|---|---|
| none | 97 | 65% |
| few | 28 | 19% |
| crowd | 25 | 17% |
| standing_nearby | 21 | 14% |
| oncoming | 16 | 11% |
| same_direction | 12 | 8% |

**Reference path and goal**

| | Scenarios | Share |
|---|---|---|
| path: straight | 81 | 54% |
| path: turn | 54 | 36% |
| path: u_turn | 15 | 10% |
| goal: ahead | 79 | 53% |
| goal: side | 51 | 34% |
| goal: behind | 20 | 13% |

**Scene type** (CLIP zero-shot on the current image; indoor and outdoor vocabularies)

| | Scenarios | Share |
|---|---|---|
| corridor | 18 | 12% |
| room | 17 | 11% |
| kitchen | 15 | 10% |
| doorway | 14 | 9% |
| road | 12 | 8% |
| dining | 11 | 7% |
| crossing | 11 | 7% |
| stairs | 10 | 7% |
| office | 9 | 6% |
| sidewalk | 7 | 5% |
| plaza | 7 | 5% |
| entrance | 7 | 5% |
| lobby | 6 | 4% |
| path | 6 | 4% |

**Conditions:** narrow passage (reference passes within 35 cm of an obstacle) in 96,
dim lighting in 55. **Difficulty:** "walk straight ahead" fails in
94 and "walk straight to the goal" in
96 of the 150 scenarios.

## Baselines

Success / collision rate / progress ratio, robot speed 0.5 m/s.

| Planner | All | Indoor | Outdoor |
|---|---|---|---|
| reference | 0.79 / 0.00 / 0.71 | 0.74 / 0.00 / 0.67 | 0.88 / 0.00 / 0.80 |
| stationary | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| straight_ahead | 0.37 / 0.42 / 0.54 | 0.23 / 0.54 / 0.47 | 0.66 / 0.18 / 0.66 |
| straight_to_goal | 0.36 / 0.64 / 1.00 | 0.20 / 0.80 / 1.00 | 0.68 / 0.32 / 1.00 |

`reference` follows the recorded path (a person driving the robot, or the simulator's shortest-path planner):
every scenario is solvable by construction, but the reference is not optimal for the goal of the prompt, so its
success is below 1. Naive planners collide often indoors: the set needs perception, not only the goal.

## How it was built

1. **Candidates, all held out from training:** CODa `coda_2hz` test + validation, JRDB `jrdb_2.5hz` (all of
   JRDB is reserved for evaluation), RoboSense `robosense_1hz` validation, HSSD `hssd_2hz` test + validation
   houses. The per-frame training configs of these sources publish only their train splits.
2. **Goal:** a point on the recorded reference path 3–10 m beyond the end of the scored horizon (the end of the
   episode for simulated ones), at least 2 m away, so the goal does not give away the answer.
3. **Validity:** the reference path, followed at 0.5 m/s, is collision-free and does not reverse; the reference
   is at most 30% over unobserved ground; nothing overlaps the robot at the start; nobody walks through a robot
   that stands still.
4. **Environment:** every real candidate not clearly outdoor by CLIP was labelled by eye (courtyards and
   covered walkways count as outdoor); simulated candidates must look indoor to CLIP with >= 0.95 and must not
   show the black background of an open sky (HSSD point-goal episodes partly run through gardens).
5. **Selection:** real scenarios first; then tag coverage, then scenarios where naive baselines fail; at most 15
   per real recording and 4 per simulated house, 5 s apart within a recording, no near-duplicate views.
6. **Audit:** every scenario was inspected on its audit image; rejected ones (with reasons) and accepted ones are
   listed in `docs/eval_audit_v1.json`, and re-running the selection reproduces this set.

## Columns

The scenario schema of the source datasets (`past_images`, `ego`, `future_tracks`, `future_lidar`,
`static_map`, `camera`, ...; see the [CODa card](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset))
plus: `suite_id`, `suite_version`, `prompt`, `environment`, `scene`, `tags`, `source`, `source_repo`,
`source_config`, `source_split`, `reference` (`teleoperated_robot` or `shortest_path_planner`), `goal_extra_m`.
`goal` is the goal of the prompt (not the end of the recorded future, unlike the source datasets).

## Limitations

- **Real indoor data is scarce.** Only 25 indoor scenarios are real (CODa, JRDB, with people); 75 are
  simulated houses without people. Report real-indoor results separately (the tags make that easy); they are indicative.
- **Open loop and non-reactive.** People in recordings do not react to the robot; the at-fault rule removes the
  worst artefacts, but interactions are not closed loop. Simulated scenarios could be run closed loop in Habitat.
- **Small.** With 150 scenarios, two planners must differ by roughly 10–15 points in collision rate to be told apart
  reliably on one binary metric; continuous scores and paired comparisons help.
- **Estimated tags.** Scene types are CLIP estimates; people tags come from the source's tracks.
- **Different robots.** Recordings come from wheeled robots (camera 0.7–0.8 m high) and a simulated agent
  (camera 1.25 m), not a humanoid: camera height and field of view vary on purpose (the suite is robot-agnostic).

## Licence

Each scenario keeps its source's licence, all non-commercial; see `LICENSE.md`:

| Source | Licence |
|---|---|
| coda | CC BY-NC-SA 4.0 (UT CODa) |
| jrdb | CC BY-NC-SA 3.0 (JRDB) |
| robosense | CC BY-NC-SA 4.0 (RoboSense) |
| hssd | CC BY-NC 4.0 (HSSD) |

Cite the source datasets: UT CODa, JRDB, RoboSense, HSSD.
