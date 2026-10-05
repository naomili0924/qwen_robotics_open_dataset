#!/usr/bin/env python
"""Score a planner on the evaluation suite (docs/eval_design.md, hnod/suite.py).

Predictions are a JSON file {suite_id: [[x, y], ...]}: a path in metres in the robot frame at the
current moment (x forward, y left), any number of points, any spacing.  The robot is assumed to
follow it at --speed m/s (0.5 by default); a path that ends early means the robot stops there.

    python scripts/evaluate_suite.py --baseline straight_to_goal
    python scripts/evaluate_suite.py --pred my_predictions.json --out results.json
    python scripts/evaluate_suite.py --data /workspace/cache/suite/suite/v1/data --baseline stationary
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import suite  # noqa: E402

REPO = "Jinyan0924/qwen_robotics_nav_eval"
GROUPS = ["env", "source", "people", "path", "goal", "scene", "layout", "lighting"]


def load_rows(args):
    from datasets import load_dataset
    if args.data and os.path.isdir(args.data):
        files = sorted(glob.glob(os.path.join(args.data, "test-*.parquet")))
        return load_dataset("parquet", data_files=files, split="train")
    return load_dataset(args.repo, args.version, split="test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--data", help="local directory of suite parquet files instead of the Hub")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--pred", help="predictions JSON {suite_id: [[x, y], ...]}")
    what.add_argument("--baseline", choices=["stationary", "straight_ahead", "straight_to_goal", "reference"])
    ap.add_argument("--speed", type=float, default=suite.ROBOT_SPEED)
    ap.add_argument("--out", help="write per-scenario and grouped results here (JSON)")
    args = ap.parse_args()

    rows = load_rows(args)
    pred = json.load(open(args.pred)) if args.pred else None
    results, missing = [], 0
    for row in rows:
        if pred is not None:
            if row["suite_id"] not in pred:
                missing += 1
                continue
            path = np.asarray(pred[row["suite_id"]], float).reshape(-1, 2)
        elif args.baseline == "reference":
            e = row["ego"]
            ref = np.stack([e["x"], e["y"]], 1)[suite.CURRENT + 1:]
            path = ref - np.array([e["x"][suite.CURRENT], e["y"][suite.CURRENT]])
        else:
            path = suite.baseline_paths(row)[args.baseline]
        r = suite.score(row, path, speed=args.speed)
        r["suite_id"], r["tags"] = row["suite_id"], list(row["tags"])
        results.append(r)
    if missing:
        print(f"warning: no prediction for {missing} of {len(rows)} scenarios; they are counted as failures")
        for _ in range(missing):
            results.append(dict(success=False, collided=False, progress_ratio=0.0, path_efficiency=0.0,
                                wiggle_rad=0.0, reference_deviation_m=np.nan, unknown_fraction=0.0, tags=[],
                                collided_dynamic=False, collided_static=False, collided_map=False,
                                collided_lidar=False))

    overall = suite.aggregate(results)
    grouped = defaultdict(dict)
    for tag in sorted({t for r in results for t in r["tags"]}):
        group = tag.split(":")[0]
        if group in GROUPS or group == "baseline_fails":
            sub = [r for r in results if tag in r["tags"]]
            grouped[group][tag] = {k: round(v, 3) for k, v in suite.aggregate(sub).items()}

    show = ["success", "collided", "progress_ratio", "path_efficiency", "wiggle_rad", "reference_deviation_m"]
    print(f"{'':32s}" + "".join(f"{k[:14]:>15s}" for k in show) + f"{'n':>6s}")
    print(f"{'all':32s}" + "".join(f"{overall[k]:15.3f}" for k in show) + f"{overall['num_scenarios']:6d}")
    for group in GROUPS:
        for tag, m in grouped.get(group, {}).items():
            print(f"{tag:32s}" + "".join(f"{m[k]:15.3f}" for k in show) + f"{m['num_scenarios']:6d}")
    if args.out:
        clean = [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for r in results]
        json.dump(dict(overall=overall, grouped=grouped, scenarios=clean, speed=args.speed), open(args.out, "w"),
                  indent=1, default=float)


if __name__ == "__main__":
    main()
