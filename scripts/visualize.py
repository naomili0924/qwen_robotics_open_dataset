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
from hnod.eval import decode_map, future_tracks  # noqa: E402
from hnod.maps import MAP_RANGE  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402

COLORS = {"PEDESTRIAN": "#d62728", "CYCLE": "#ff7f0e", "VEHICLE": "#9467bd", "OTHER_MOVABLE": "#8c564b",
          "STATIC": "#1f77b4"}


def box_corners(x, y, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    local = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


LIDAR_COLORS = np.array([[0.75, 0.75, 0.75], [0.1, 0.1, 0.1], [0.84, 0.15, 0.16]])  # ground, static, dynamic


def draw(ax, row, view=None, lidar_step=None):
    """Bird's-eye view with +x (ego forward) up and +y (left) to the left, matching the map image.

    lidar_step: if set, draw the lidar points of that future step (0 = current) instead of the static map.
    """
    if lidar_step is not None and row.get("future_lidar") is not None:
        pts = row["future_lidar"]
        x, y = np.array(pts["x"][lidar_step]) / 100, np.array(pts["y"][lidar_step]) / 100
        ax.scatter(y, x, s=0.3, c=LIDAR_COLORS[np.array(pts["label"][lidar_step])], linewidths=0)
        ax.set_facecolor("white")
    elif row.get("static_map") is not None:
        img = decode_map(row["static_map"])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=[MAP_RANGE, -MAP_RANGE, -MAP_RANGE, MAP_RANGE], alpha=0.6)
    tr = future_tracks(row)
    for n in range(len(tr["id"])):
        valid = np.array(tr["valid"][n])
        x, y, h = (np.array(tr[k][n], dtype=float) for k in ("x", "y", "heading"))
        col = "#7f7f7f" if tr["is_operator"][n] else COLORS[tr["object_type"][n]]
        ref = int(np.argmax(valid))  # the current step if the object is there, else its first sighting
        ax.add_patch(Polygon(box_corners(x[ref], y[ref], h[ref], tr["length"][n], tr["width"][n])[:, ::-1],
                             closed=True, fill=bool(valid[0]), alpha=0.5 if valid[0] else 0.9,
                             facecolor=col, edgecolor=col, lw=0.8))
        if not tr["is_stationary"][n]:
            ax.plot(y[valid], x[valid], ".-", color=col, lw=1, ms=3)
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


def draw_image(ax, row, index=-1):
    """One of the past images (default: the current step) with the recorded future path projected in."""
    img = np.asarray(row["past_images"][index])
    ax.imshow(img)
    cam, ego = row["camera"], row["ego"]
    K = np.array(cam["K"]).reshape(3, 3)
    cam_from_scn = np.linalg.inv(np.array(cam["T_scenario_from_camera"][index]).reshape(4, 4))
    path = np.stack([ego["x"], ego["y"], np.zeros(len(ego["x"])), np.ones(len(ego["x"]))], 1)[CURRENT + 1:]
    pc = path @ cam_from_scn.T
    front = pc[:, 2] > 0.3
    uv = (pc[front, :3] @ K.T)
    uv = uv[:, :2] / uv[:, 2:]
    ax.plot(uv[:, 0], uv[:, 1], ".-", color="#2ca02c", lw=2, ms=6)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title("current image, recorded future path on the ground", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with <split>-*.parquet files")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--view", type=float, default=None, help="half-width of the plotted area [m]")
    ap.add_argument("--panels", action="store_true",
                    help="one row per scenario: current image, static map, lidar points of the last future step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("parquet", data_files=sorted(glob.glob(f"{args.data}/{args.split}-*.parquet")), split="train")
    if args.ids:
        ids = ds["scenario_id"]
        pick = [ids.index(i) for i in args.ids]
    else:
        pick = np.random.default_rng(args.seed).choice(len(ds), size=min(args.num, len(ds)), replace=False)
    if args.panels:
        fig, axs = plt.subplots(len(pick), 3, figsize=(21, 6.4 * len(pick)), squeeze=False,
                                gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
        for r, i in enumerate(pick):
            row = ds[int(i)]
            draw_image(axs[r, 0], row)
            draw(axs[r, 1], row, args.view)
            draw(axs[r, 2], row, args.view, lidar_step=len(row["future_lidar"]["x"]) - 1)
            axs[r, 2].set_title("lidar at the last future step: ground / static / dynamic", fontsize=9)
    else:
        cols = min(3, len(pick))
        rows = int(np.ceil(len(pick) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
        for ax, i in zip(axs.ravel(), pick):
            draw(ax, ds[int(i)], args.view)
        for ax in axs.ravel()[len(pick):]:
            ax.axis("off")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=70)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
