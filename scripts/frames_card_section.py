#!/usr/bin/env python
"""Add the per-frame configs (frames / episodes, hnod.frames) to an existing scenario dataset card.

Reads the repo's episodes config for statistics, adds the two configs to the card's YAML header and
inserts (or replaces) a "Per-frame training configs" section before the first "## " heading after
the introduction.  Then upload the card as README.md.

    python scripts/frames_card_section.py --repo Jinyan0924/qwen_robotics_open_dataset --card dataset_card.md \
        --note "Built from coda_10hz train: 10 Hz frames."
"""
import argparse
import re

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

BEGIN, END = "<!-- per-frame:begin -->", "<!-- per-frame:end -->"


def stats(repo):
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    splits = sorted({f.split("/")[-1].split("-")[0] for f in files if f.startswith("data/episodes/")})
    out = {}
    for sp in splits:
        e = load_dataset(repo, data_files={sp: f"data/episodes/{sp}-*.parquet"}, split=sp)
        prob = np.concatenate([np.asarray(p, float) for p in e["frame_indoor_prob"]]) if len(e) else np.zeros(0)
        out[sp] = dict(episodes=len(e), frames=int(sum(e["num_frames"])), hours=sum(e["duration_s"]) / 3600,
                       km=sum(e["path_length_m"]) / 1000, indoor=float((prob > 0.5).mean()) if len(prob) else 0.0,
                       speed=float(np.median(e["median_speed_mps"])), rate=float(np.median(e["rate_hz"])),
                       env=dict(zip(*np.unique(e["environment"], return_counts=True))),
                       method=e[0]["environment_method"] if len(e) else "")
    return out


def section(repo, st, note):
    rows = "\n".join(f"| {sp} | {s['episodes']:,} | {s['frames']:,} | {s['rate']:g} Hz | {s['hours']:.1f} | {s['km']:.1f} | "
                     f"{100 * s['indoor']:.0f}% | {s['speed']:.2f} m/s |" for sp, s in st.items())
    method = next(iter(st.values()))["method"] if st else ""
    return f"""{BEGIN}
## Per-frame training configs: `frames` and `episodes`

The same recordings in the per-frame training format of
[qwen_robotics_open_dataset](https://github.com/naomili0924/qwen_robotics_open_dataset) (described in full on the
[EgoWalk card](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_egowalk)): one row per
camera frame with the raw metric pose (x, y, z, yaw), plus one row per episode with camera, environment and
embodiment. Training samples (history, waypoints by distance along the path, a goal beyond the horizon, a text
prompt) are cut at load time with `hnod.windows.FrameWindows`.

| Split | Episodes | Frames | Rate | Hours | km | Indoor frames | Median speed |
|---|---|---|---|---|---|---|---|
{rows}

{note} Frames were collected from the scenario rows (each imaged frame once) and their poses mapped back to the
source's world frame; the last second or so of each recorded segment, which has poses but no images in the
scenario rows, is dropped. Indoor / outdoor: `{method}`, a rough estimate: CLIP tends to call courtyards and
covered walkways indoor (the evaluation suite's labels were reviewed by eye instead). **Only the train split is published here: the
held-out splits of this source are part of the evaluation suite
([qwen_robotics_nav_eval](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_nav_eval)) and must not be
trained or tuned on.**

```python
from datasets import load_dataset
from hnod.windows import FrameWindows
frames = load_dataset("{repo}", "frames", split="train")
episodes = load_dataset("{repo}", "episodes", split="train")
samples = FrameWindows(frames, episodes)
```
{END}
"""


def patch_card(text, sec, splits):
    yaml_add = "- config_name: frames\n  data_files:\n" + "".join(
        f"  - split: {sp}\n    path: data/frames/{sp}-*.parquet\n" for sp in splits) + \
        "- config_name: episodes\n  data_files:\n" + "".join(
        f"  - split: {sp}\n    path: data/episodes/{sp}-*.parquet\n" for sp in splits)
    head_end = text.index("\n---", 3)
    head, body = text[:head_end], text[head_end:]
    head = re.sub(r"- config_name: frames\n(  .*\n)*- config_name: episodes\n(  .*\n?)*", "", head + "\n").rstrip("\n")
    head = head + "\n" + yaml_add.rstrip("\n")
    if BEGIN in body:
        body = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END) + "\n", sec, body, flags=re.S)
    else:
        first = body.index("\n## ")
        body = body[:first + 1] + sec + "\n" + body[first + 1:]
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    st = stats(args.repo)
    text = patch_card(open(args.card).read(), section(args.repo, st, args.note), list(st))
    open(args.card, "w").write(text)
    print({k: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in s.items()} for k, s in st.items()})
    if args.upload:
        HfApi().upload_file(path_or_fileobj=args.card, path_in_repo="README.md", repo_id=args.repo,
                            repo_type="dataset", commit_message="Card: per-frame training configs")


if __name__ == "__main__":
    main()
