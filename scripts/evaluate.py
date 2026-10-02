#!/usr/bin/env python
"""Score predicted ego trajectories (or a built-in baseline) for collisions.

Predictions file: JSON mapping scenario_id -> [[x, y], ...] with one position
per future step (10), in the scenario frame (ego at the origin facing +x at the
current step).

Usage:
    python scripts/evaluate.py --data data/hf/coda_2hz --split test --baseline constant_velocity
    python scripts/evaluate.py --data data/hf/coda_2hz --split test --pred my_predictions.json --radius 0.3
    python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz --split test --baseline expert
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import eval as ev  # noqa: E402

_ctx = {}


def _score(i):
    row = _ctx["ds"][i]
    if _ctx["pred"] is not None:
        if row["scenario_id"] not in _ctx["pred"]:
            return None
        pred = _ctx["pred"][row["scenario_id"]]
    else:
        pred = ev.BASELINES[_ctx["baseline"]](row)
    res = ev.evaluate_scenario(row, pred, radius=_ctx["radius"], use_map=not _ctx["no_map"])
    res["num_pedestrians"] = row["num_pedestrians"]
    return res


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local directory with <split>-*.parquet files")
    src.add_argument("--repo", help="Hugging Face dataset repo id")
    ap.add_argument("--config", help="dataset config when using --repo, e.g. coda_2hz")
    ap.add_argument("--split", default="test")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--pred", help="JSON file with predictions")
    what.add_argument("--baseline", choices=sorted(ev.BASELINES))
    ap.add_argument("--radius", type=float, default=ev.DEFAULT_RADIUS, help="agent radius [m]")
    ap.add_argument("--no-map", action="store_true", help="ignore the lidar static map")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", help="write per-scenario results to this JSON file")
    args = ap.parse_args()

    if args.data:
        ds = load_dataset("parquet", data_files=sorted(glob.glob(f"{args.data}/{args.split}-*.parquet")), split="train")
    else:
        ds = load_dataset(args.repo, args.config, split=args.split)
    _ctx.update(ds=ds, pred=json.load(open(args.pred)) if args.pred else None, baseline=args.baseline,
                radius=args.radius, no_map=args.no_map)
    with ProcessPoolExecutor(args.workers) as ex:  # fork: workers inherit _ctx
        results = [r for r in ex.map(_score, range(len(ds)), chunksize=16) if r is not None]
    if not results:
        sys.exit("no scenario ids in the prediction file match this split")
    if args.pred and len(results) < len(ds):
        print(f"warning: predictions cover {len(results)} of {len(ds)} scenarios; metrics are over those only")

    agg = ev.aggregate(results)
    print(json.dumps(agg, indent=2))
    if args.out:
        clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
                 for r in results]
        json.dump(dict(summary=agg, scenarios=clean), open(args.out, "w"))


if __name__ == "__main__":
    main()
