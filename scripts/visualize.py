#!/usr/bin/env python
"""Render scenarios to PNG for inspection.

Usage:
    python scripts/visualize.py --data data/hf/coda_2hz --split test --num 6 --out figures/coda_2hz_test.png
    python scripts/visualize.py --data data/hf/coda_2hz --split test --ids coda_01_001800_2hz --out one.png
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod.eval import decode_map  # noqa: E402
from hnod.maps import MAP_RANGE  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402

COLORS = {"PEDESTRIAN": "#d62728", "CYCLE": "#ff7f0e", "VEHICLE": "#9467bd", "OTHER_MOVABLE": "#8c564b",
          "STATIC": "#1f77b4"}


def box_corners(x, y, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    local = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def draw(ax, row, view=None, map_key="static_map"):
    """Plot with +x (ego forward) up and +y (left) to the left, matching the map image."""
    img = decode_map(row[map_key])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=[MAP_RANGE, -MAP_RANGE, -MAP_RANGE, MAP_RANGE], alpha=0.6)
    tr = row["tracks"]
    for n in range(len(tr["id"])):
        valid = np.array(tr["valid"][n])
        x, y, h = (np.array(tr[k][n], dtype=float) for k in ("x", "y", "heading"))
        col = "#7f7f7f" if tr["is_operator"][n] else COLORS[tr["object_type"][n]]
        ref = CURRENT if valid[CURRENT] else int(np.flatnonzero(valid)[np.argmin(np.abs(np.flatnonzero(valid) - CURRENT))])
        ax.add_patch(Polygon(box_corners(x[ref], y[ref], h[ref], tr["length"][n], tr["width"][n])[:, ::-1],
                             closed=True, fill=valid[CURRENT], alpha=0.5 if valid[CURRENT] else 0.9,
                             facecolor=col, edgecolor=col, lw=0.8))
        if not tr["is_stationary"][n]:
            past, fut = valid & (np.arange(len(valid)) <= CURRENT), valid & (np.arange(len(valid)) >= CURRENT)
            ax.plot(y[past], x[past], "-", color=col, lw=1, alpha=0.5)
            ax.plot(y[fut], x[fut], ".-", color=col, lw=1, ms=3)
    ego = row["ego"]
    ex, ey = np.array(ego["x"]), np.array(ego["y"])
    ax.plot(ey[:CURRENT + 1], ex[:CURRENT + 1], "-", color="#2ca02c", lw=2, alpha=0.5)
    ax.plot(ey[CURRENT:], ex[CURRENT:], ".-", color="#2ca02c", lw=2, ms=5)
    ax.add_patch(Polygon(box_corners(0, 0, 0, ego["length"], ego["width"])[:, ::-1], closed=True,
                         facecolor="#2ca02c", edgecolor="k", lw=0.8))
    ax.plot(row["goal"][1], row["goal"][0], "*", color="gold", mec="k", ms=14)
    view = view or MAP_RANGE
    ax.set_xlim(view, -view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.set_title(f"{row['scenario_id']}  ({row['num_pedestrians']} peds)", fontsize=9)
    ax.set_xlabel("y [m] (left +)")
    ax.set_ylabel("x [m] (forward +)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with <split>-*.parquet files")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--view", type=float, default=None, help="half-width of the plotted area [m]")
    ap.add_argument("--map", default="static_map", choices=["static_map", "observed_map"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("parquet", data_files=sorted(glob.glob(f"{args.data}/{args.split}-*.parquet")), split="train")
    if args.ids:
        ids = ds["scenario_id"]
        pick = [ids.index(i) for i in args.ids]
    else:
        pick = np.random.default_rng(args.seed).choice(len(ds), size=min(args.num, len(ds)), replace=False)
    cols = min(3, len(pick))
    rows = int(np.ceil(len(pick) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
    for ax, i in zip(axs.ravel(), pick):
        draw(ax, ds[int(i)], args.view, args.map)
    for ax in axs.ravel()[len(pick):]:
        ax.axis("off")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=70)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
