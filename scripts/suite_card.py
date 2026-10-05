#!/usr/bin/env python
"""Dataset card, distribution figure and baseline table for a written evaluation suite.

    python scripts/suite_card.py --suite /workspace/cache/suite/suite/v1 --out-card dataset_card_eval.md
"""
import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import suite  # noqa: E402

GROUPS = [("env", "Environment"), ("source", "Source"), ("scene", "Scene type (CLIP estimate)"),
          ("people", "People (recorded tracks)"), ("path", "Reference path"), ("goal", "Goal bearing"),
          ("layout", "Layout"), ("lighting", "Lighting"), ("baseline_fails", "Naive baselines that fail")]
BASELINES = ["reference", "stationary", "straight_ahead", "straight_to_goal"]


def distribution(rows):
    tags = Counter(t for r in rows for t in r["tags"])
    out = {}
    for g, _ in GROUPS:
        out[g] = {t.split(":", 1)[1]: n for t, n in sorted(tags.items(), key=lambda kv: -kv[1]) if t.split(":")[0] == g}
    return out


def plot_distribution(dist, n, path):
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (g, title) in zip(axs.ravel(), GROUPS):
        d = dist.get(g, {})
        ax.barh(list(d)[::-1], list(d.values())[::-1], color="#4c72b0")
        for y, v in enumerate(list(d.values())[::-1]):
            ax.text(v, y, f" {v}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(d.values(), default=1) * 1.2)
    fig.suptitle(f"Scenario distribution ({n} scenarios; a scenario can carry several people / baseline tags)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def baseline_table(rows):
    out = {}
    for b in BASELINES:
        res = []
        for r in rows:
            if b == "reference":
                e = r["ego"]
                ref = np.stack([e["x"], e["y"]], 1)[suite.CURRENT + 1:]
                p = ref - np.array([e["x"][suite.CURRENT], e["y"][suite.CURRENT]])
            else:
                p = suite.baseline_paths(r)[b]
            x = suite.score(r, p)
            x["env"] = r["environment"]
            res.append(x)
        out[b] = {env: suite.aggregate([x for x in res if env in ("all", x["env"])]) for env in ("all", "indoor", "outdoor")}
    return out


def example_grid(audit_dir, ids, path):
    import matplotlib.image as mpimg
    fig, axs = plt.subplots(len(ids), 1, figsize=(15, 6.5 * len(ids)))
    for ax, sid in zip(np.atleast_1d(axs), ids):
        ax.imshow(mpimg.imread(os.path.join(audit_dir, f"{sid}.png")))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


LICENCES = {"coda": "CC BY-NC-SA 4.0 (UT CODa)", "jrdb": "CC BY-NC-SA 3.0 (JRDB)",
            "robosense": "CC BY-NC-SA 4.0 (RoboSense)", "hssd": "CC BY-NC 4.0 (HSSD)", "musohu": "CC0 (MuSoHu)"}
HORIZON = {"coda": "5 s", "hssd": "5 s", "musohu": "5 s", "jrdb": "4 s", "robosense": "10 s"}
NAMES = {"coda": "UT CODa", "jrdb": "JRDB", "robosense": "RoboSense", "hssd": "HSSD", "musohu": "MuSoHu"}


def _table(d, n):
    rows = "\n".join(f"| {k} | {v} | {100 * v / n:.0f}% |" for k, v in d.items())
    return "| | Scenarios | Share |\n|---|---|---|\n" + rows


def write_card(path, dist, table, n, by_env_source, version, repo):
    t = table
    srcs = sorted({src for _, src in by_env_source})
    lic = "\n".join(f"| {k} | {LICENCES[k]} |" for k in srcs)
    hor = ", ".join(f"{HORIZON[k]} for {NAMES[k]}" for k in srcs)
    real_in = sum(k for (env, src), k in by_env_source.items() if env == "indoor" and src != "hssd")
    sim_note = "" 
    sim_in = sum(k for (env, src), k in by_env_source.items() if env == "indoor" and src == "hssd")
    sim_note = f"; {sim_in} are simulated houses without people (report real-indoor results separately)" if sim_in else ""
    base_rows = "\n".join(
        f"| {b} | " + " | ".join(f"{t[b][env]['success']:.2f} / {t[b][env]['collided']:.2f} / {t[b][env]['progress_ratio']:.2f}"
                                 for env in ("all", "indoor", "outdoor")) + " |" for b in BASELINES)
    comp = "\n".join(f"| {env} | {src} | {k} |" for (env, src), k in sorted(by_env_source.items()))
    text = f"""---
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
- config_name: {version}
  data_files:
  - split: test
    path: data/{version}/test-*.parquet
---

# Robot navigation evaluation suite ({version})

{n} hand-audited scenarios for **image-in, path-out navigation policies of an indoor robot** (a walking
humanoid, about 0.5 m/s): 100 indoor and 50 outdoor, drawn from {len(srcs)} open datasets, each with a text prompt
that contains the goal and 3D ground truth (lidar, tracked people, obstacle maps) for scoring. Built by
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset)
(`scripts/build_eval_suite.py`; design and decisions in `docs/eval_design.md`, every audit decision in
`docs/eval_audit_{version}.json`).

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
  ({hor}); a path that ends early means it stops there.
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
python scripts/evaluate_suite.py --pred my_predictions.json     # {{"{version}-0000": [[x, y], ...], ...}}
python scripts/evaluate_suite.py --baseline straight_to_goal
python -m vla.predict_suite --checkpoint runs/<run>/last --out preds.json   # a policy trained with vla/
```

## Scenario distribution

![distribution](figures/eval_distribution.png)

| Environment | Source | Scenarios |
|---|---|---|
{comp}

**Environment**

{_table(dist["env"], n)}

**People around the robot** (from the recorded tracks; a scenario can have several interaction tags)

{_table(dist["people"], n)}

**Reference path and goal**

{_table({**{f"path: {k}": v for k, v in dist["path"].items()}, **{f"goal: {k}": v for k, v in dist["goal"].items()}}, n)}

**Scene type** (CLIP zero-shot on the current image; indoor and outdoor vocabularies)

{_table(dist["scene"], n)}

**Conditions:** narrow passage (reference passes within 35 cm of an obstacle) in {dist["layout"].get("narrow", 0)},
dim lighting in {dist["lighting"].get("dim", 0)}. **Difficulty:** "walk straight ahead" fails in
{dist["baseline_fails"].get("straight_ahead", 0)} and "walk straight to the goal" in
{dist["baseline_fails"].get("straight_to_goal", 0)} of the {n} scenarios.

## Baselines

Success / collision rate / progress ratio, robot speed 0.5 m/s.

| Planner | All | Indoor | Outdoor |
|---|---|---|---|
{base_rows}

`reference` follows the recorded path (a person driving the robot, or the simulator's shortest-path planner):
every scenario is solvable by construction, but the reference is not optimal for the goal of the prompt, so its
success is below 1. Naive planners collide often indoors: the set needs perception, not only the goal.

## How it was built

1. **Candidates, all held out from training:** CODa `coda_2hz` test + validation, JRDB `jrdb_2.5hz` (all of
   JRDB is reserved for evaluation), RoboSense `robosense_1hz` validation, HSSD `hssd_2hz` test + validation
   houses, and MuSoHu indoor walks (all of it reserved; people tracked in its lidar, not annotated). The per-frame
   training configs of these sources publish only their train splits.
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
   listed in `docs/eval_audit_{version}.json`, and re-running the selection reproduces this set.

## Columns

The scenario schema of the source datasets (`past_images`, `ego`, `future_tracks`, `future_lidar`,
`static_map`, `camera`, ...; see the [CODa card](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset))
plus: `suite_id`, `suite_version`, `prompt`, `environment`, `scene`, `tags`, `source`, `source_repo`,
`source_config`, `source_split`, `reference` (`teleoperated_robot` or `shortest_path_planner`), `goal_extra_m`.
`goal` is the goal of the prompt (not the end of the recorded future, unlike the source datasets).

## Limitations

- **Indoor scenes:** {real_in} of the 100 indoor scenarios are real recordings{sim_note}.
  MuSoHu people are tracked in its lidar, not annotated: missed or spurious people are possible.
- **Open loop and non-reactive.** People in recordings do not react to the robot; the at-fault rule removes the
  worst artefacts, but interactions are not closed loop. Simulated scenarios could be run closed loop in Habitat.
- **Small.** With {n} scenarios, two planners must differ by roughly 10–15 points in collision rate to be told apart
  reliably on one binary metric; continuous scores and paired comparisons help.
- **Estimated tags.** Scene types are CLIP estimates; people tags come from the source's tracks.
- **Different embodiments.** Recordings come from wheeled robots (camera 0.7–0.8 m high), a walking person's
  helmet (MuSoHu, about 1.7 m) and a simulated agent: camera height and field of view vary on purpose.

## Licence

Each scenario keeps its source's licence, all non-commercial; see `LICENSE.md`:

| Source | Licence |
|---|---|
""" + lic + """

Cite the source datasets: """ + ", ".join(NAMES[k] for k in srcs) + """.
"""
    open(path, "w").write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="written suite directory (data/, audit/)")
    ap.add_argument("--out-card", default="dataset_card_eval.md")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--examples", nargs="*", default=[])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_eval")
    args = ap.parse_args()
    rows = list(load_dataset("parquet", data_files=sorted(glob.glob(f"{args.suite}/data/test-*.parquet")),
                             split="train"))
    dist = distribution(rows)
    Path(args.figures).mkdir(exist_ok=True)
    plot_distribution(dist, len(rows), f"{args.figures}/eval_distribution.png")
    if args.examples:
        example_grid(f"{args.suite}/audit", args.examples, f"{args.figures}/eval_examples.png")
    table = baseline_table(rows)
    import json
    json.dump(dict(distribution=dist, baselines=table), open(f"{args.suite}/summary.json", "w"), indent=1,
              default=float)
    print(json.dumps(dict(distribution=dist), indent=1))
    by_env_source = Counter((r["environment"], r["source"]) for r in rows)
    write_card(args.out_card, dist, table, len(rows), by_env_source, args.version, args.repo)
    for b, t in table.items():
        print(b, {env: {k: round(v, 3) for k, v in m.items() if k in ("success", "collided", "progress_ratio",
                                                                         "reference_deviation_m")} for env, m in t.items()})


if __name__ == "__main__":
    main()
