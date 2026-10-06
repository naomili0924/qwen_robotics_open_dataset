#!/usr/bin/env python
"""Figures and dataset card for the deduplicated pretraining set.

    python scripts/dedup_card.py --selected /dev/shm/dedup/selected --published /dev/shm/dedup/publish --upload
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKETS = ["straight", "standing", "slowing", "speeding_up", "turn_left", "turn_right", "sharp_left", "sharp_right",
           "stopping", "starting", "weaving"]
SHORT = {"qwen_robotics_open_dataset_egowalk": "EgoWalk", "habitat_hssd_pointgoal_nav_scenarios": "HSSD",
         "qwen_robotics_open_dataset_robosense": "RoboSense", "qwen_robotics_open_dataset": "CODa"}
COL = {"EgoWalk": "#2a78d6", "HSSD": "#eb6834", "RoboSense": "#1baf7a", "CODa": "#eda100"}


def fig_buckets(summary, path):
    srcs = list(summary["buckets"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    y = np.arange(len(BUCKETS))
    for ax, key, title in zip(axs, ("before", "after"), ("All candidate samples (one per second)", "Kept: motion-diverse, then scene-diverse")):
        left = np.zeros(len(BUCKETS))
        for s in srcs:
            v = np.array([summary["buckets"][s].get(b, {}).get(key, 0) for b in BUCKETS])
            ax.barh(y, v, left=left, color=COL[SHORT[s]], label=SHORT[s], height=0.7)
            left += v
        for i, tot in enumerate(left):
            ax.text(tot, i, f" {int(tot):,}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(BUCKETS)
        ax.invert_yaxis()
        ax.set_title(f"{title}: {summary[key]:,}", fontsize=10)
        ax.set_xlim(0, left.max() * 1.18)
        ax.grid(axis="x", alpha=.3)
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Motion classes of the next 5 s, before and after deduplication", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_scene_map(selected, path):
    sel = pd.read_parquet(f"{selected}/samples.parquet")
    E_after = np.load(f"{selected}/embeddings_kept.npy").astype(np.float32)
    E_before = np.load(f"{selected}/embeddings_sample_before.npy").astype(np.float32)
    mu = E_before.mean(0)
    _, _, vt = np.linalg.svd(E_before[:5000] - mu, full_matrices=False)
    P = vt[:2]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    xy = (E_before - mu) @ P.T
    axs[0].scatter(xy[:, 0], xy[:, 1], s=2, c="#8a9399", alpha=.4)
    axs[0].set_title(f"Before: {len(E_before):,} random candidates (scene embedding, 2 principal axes)", fontsize=10)
    xy = (E_after - mu) @ P.T
    src = sel["repo"].str.split("/").str[1].map(SHORT)
    for s in COL:
        m = (src == s).to_numpy()
        axs[1].scatter(xy[m, 0], xy[m, 1], s=2, c=COL[s], alpha=.5, label=f"{s} ({m.sum():,})")
    axs[1].set_title(f"After: all {len(E_after):,} kept samples", fontsize=10)
    axs[1].legend(fontsize=8, markerscale=5)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(axs[0].get_xlim())
        ax.set_ylim(axs[0].get_ylim())
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_examples(published, path, per_bucket=4, seed=0):
    """Current view + the 5 s trajectory for a few kept samples of every motion class."""
    import io
    from PIL import Image
    samples = pq.read_table(f"{published}/samples/train-00000.parquet").to_pandas()
    rng = np.random.default_rng(seed)
    picks = []
    for b in BUCKETS:
        sub = samples[samples["bucket"] == b]
        if len(sub):
            picks += [(b, r) for r in sub.iloc[rng.choice(len(sub), min(per_bucket, len(sub)), replace=False)].itertuples()]
    want = {(r.episode_id, int(r.frame_index)) for _, r in picks}
    found = {}
    for f in sorted(glob.glob(f"{published}/frames/train-*.parquet")):
        t = pq.read_table(f, columns=["episode_id", "frame_index", "timestamp", "pose", "image"])
        eps = set(t.column("episode_id").to_pylist())
        if not any(e in eps for e, _ in want):
            continue
        d = t.to_pandas()
        for e, fi in [k for k in want if k[0] in eps]:
            ep = d[d["episode_id"] == e].sort_values("frame_index")
            row = ep[ep["frame_index"] == fi].iloc[0]
            poses = np.stack(ep["pose"].to_numpy())
            tt = ep["timestamp"].to_numpy()
            times = row["timestamp"] + 0.5 * np.arange(1, 11)
            x = np.interp(times, tt, poses[:, 0]) - row["pose"][0]
            y = np.interp(times, tt, poses[:, 1]) - row["pose"][1]
            c, s = np.cos(row["pose"][3]), np.sin(row["pose"][3])
            found[(e, fi)] = (Image.open(io.BytesIO(row["image"]["bytes"])), np.c_[c * x + s * y, -s * x + c * y])
    n = len(picks)
    cols = per_bucket * 2
    rows_n = (n + per_bucket - 1) // per_bucket
    fig, axs = plt.subplots(rows_n, cols, figsize=(2.2 * cols, 2.0 * rows_n))
    for i, (b, r) in enumerate(picks):
        rr, cc = divmod(i, per_bucket)
        ax_img, ax_tr = axs[rr, 2 * cc], axs[rr, 2 * cc + 1]
        if (r.episode_id, int(r.frame_index)) in found:
            img, tr = found[(r.episode_id, int(r.frame_index))]
            ax_img.imshow(img)
            ax_tr.plot(np.r_[0, -tr[:, 1]], np.r_[0, tr[:, 0]], "o-", ms=2, c="#1baf7a")
            L = max(1.0, np.abs(tr).max() * 1.1)
            ax_tr.set_xlim(-L, L)
            ax_tr.set_ylim(-L * 0.3, L * 1.7)
            ax_tr.set_aspect("equal")
            ax_tr.grid(alpha=.3)
        ax_img.set_title(f"{b} · {SHORT.get(r.source, r.source)}", fontsize=7)
        ax_img.axis("off")
        ax_tr.tick_params(labelsize=5)
    for ax in axs.ravel()[len(picks) * 2:]:
        ax.axis("off")
    fig.suptitle("Kept samples by motion class: current view and the recorded next 5 s (forward is up)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def card(summary, totals, selected_args):
    srcs = summary["per_source"]
    rows = "\n".join(f"| {SHORT[s]} | {v['before']:,} | {v['after']:,} | {100 * v['after'] / max(1, v['before']):.0f}% |" for s, v in srcs.items())
    brow = "\n".join(f"| {b} | " + " | ".join(f"{summary['buckets'][s].get(b, {}).get('after', 0):,}" for s in srcs)
                     + f" | {sum(summary['buckets'][s].get(b, {}).get('after', 0) for s in srcs):,} |" for b in BUCKETS)
    return f"""---
license: other
license_name: mixed-non-commercial
task_categories:
- robotics
tags:
- navigation
- egocentric
- trajectory-prediction
- deduplicated
pretty_name: Navigation pretraining frames, deduplicated
size_categories:
- 100K<n<1M
configs:
- config_name: frames
  default: true
  data_files:
  - split: train
    path: data/frames/train-*.parquet
- config_name: episodes
  data_files:
  - split: train
    path: data/episodes/train-*.parquet
- config_name: samples
  data_files:
  - split: train
    path: data/samples/train-*.parquet
---

# Navigation pretraining frames, deduplicated

A **{summary['after']:,}-sample** training set for the final-frame pretraining task of
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset): past camera views + the view
5 s ahead → the recorded motion in between. It is cut from the same four open sources as the full per-frame sets
(1.11 M frames), keeping **motion diversity first and scene diversity second**, so that a pass over it costs a
fraction of the time and the rare behaviours (turns, stops, starts) are not drowned by straight walking.

![motion classes](figures/buckets.png)

![examples](figures/examples.png)

## How it was built

1. **Candidates:** one sample per second of every training recording (`scripts/build_dedup_index.py`): the next
   5 s of recorded motion is summarised (mean speed, speed at the start and end, net heading change, straightness)
   and the current view is embedded with CLIP ViT-L/14.
2. **Motion class** of each candidate: {", ".join(BUCKETS)} (`bucket()` in the same script).
3. **Selection** (`scripts/select_dedup.py`), per source and motion class: rare classes are kept up to
   {selected_args['cap_rare']:,} each, `straight` is capped at {selected_args['cap_straight']:,} and `standing` at
   {selected_args['cap_standing']:,}; inside a class, candidates are visited in random order and one is skipped when its
   scene embedding is more than {selected_args['sim']:.2f} cosine-similar to a kept sample of the same class (same place,
   same view, same motion = duplicate). Simulated houses: at most {selected_args['per_house']} samples per house.
4. **Publication** (`scripts/publish_dedup.py`): the per-frame format of the full sets, restricted to episodes with
   kept samples; images (long side {selected_args['max_side']} px) only for the frames a kept sample needs.

| Source | Candidates | Kept | Share |
|---|---|---|---|
{rows}
| **Total** | **{summary['before']:,}** | **{summary['after']:,}** | **{100 * summary['after'] / summary['before']:.0f}%** |

Kept samples per motion class:

| Class | {" | ".join(SHORT[s] for s in srcs)} | Total |
|---|{"---|" * len(srcs)}---|
{brow}

![scene map](figures/scene_map.png)

*Scene embeddings projected on two axes: the kept samples cover the same regions as the candidates with far fewer
points; the simulated houses (orange) collapse from a dense blob to a thin layer.*

## Contents

- `frames` ({totals['frames']:,} rows, {totals['images']:,} with an image): `episode_id`, `frame_index`, `source_frame`,
  `timestamp`, `image` (None where no kept sample needs it), `image_right` (None), `pose` (x, y, z, yaw).
- `episodes` ({totals['episodes']:,} rows): the source episode tables (embodiment, rate, camera, environment, ...).
- `samples` ({summary['after']:,} rows): `episode_id`, `frame_index`, `source`, `embodiment`, `bucket` and the motion
  summary of the next 5 s.

## Use

```bash
bash scripts/run_ff.sh <run> <steps> 2b --frames-repos Jinyan0924/qwen_robotics_nav_pretrain_dedup
```

The streaming loader reads the `samples` table and cuts only the listed samples; sources are already balanced
inside the selection, so `--frames-mix` has no effect with a single repo.

## Limits

- Deduplication is by appearance of the current view and by the motion class; two walks through the same corridor
  with the same motion are kept once even if the people around differed.
- The motion classes are thresholds on 5 s of recorded motion (20° and 60° of heading change, 0.15 m/s for
  standing, 0.3 m/s speed change); they describe the recording, not an intent.
- Licences follow the sources: EgoWalk MIT, HSSD CC BY-NC 4.0, RoboSense and CODa CC BY-NC-SA 4.0 (non-commercial).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="/dev/shm/dedup/selected")
    ap.add_argument("--published", default="/dev/shm/dedup/publish")
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_pretrain_dedup")
    ap.add_argument("--figures", default="figures/dedup")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    summary = json.load(open(f"{args.selected}/summary.json"))
    totals = json.load(open(f"{args.published}/totals.json"))
    sel_args = json.load(open(f"{args.selected}/args.json"))
    Path(args.figures).mkdir(parents=True, exist_ok=True)
    fig_buckets(summary, f"{args.figures}/buckets.png")
    fig_scene_map(args.selected, f"{args.figures}/scene_map.png")
    fig_examples(args.published, f"{args.figures}/examples.png")
    text = card(summary, totals, sel_args)
    open("dataset_card_dedup.md", "w").write(text)
    if args.upload:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.figures, path_in_repo="figures",
                          commit_message="Figures")
        api.upload_file(path_or_fileobj="dataset_card_dedup.md", path_in_repo="README.md", repo_id=args.repo,
                        repo_type="dataset", commit_message="Dataset card")
        print("https://huggingface.co/datasets/" + args.repo)


if __name__ == "__main__":
    main()
