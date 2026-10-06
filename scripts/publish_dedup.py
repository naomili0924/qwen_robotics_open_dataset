#!/usr/bin/env python
"""Pass 3 of the deduplication: publish the selected samples as a self-contained per-frame dataset.

For every episode with at least one kept sample the `frames` table keeps every frame's pose (the targets are
read from poses at full rate) but stores an image only for the frames a kept sample needs: its 5 past views,
the current view and the final view (long side resized to --max-side).  The `episodes` table is copied, and a
`samples` table lists the kept samples with their motion class.  Shards hold whole episodes, so the streaming
loader works unchanged with `--frames-repos <this repo>`; it reads the sample list and cuts only those.

    python scripts/publish_dedup.py --selected /dev/shm/dedup/selected --repo Jinyan0924/qwen_robotics_nav_pretrain_dedup --upload
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import frames as fr  # noqa: E402
from vla.stream import _Source, read_shard  # noqa: E402

SAMPLE_FEATURES = Features({"episode_id": Value("string"), "frame_index": Value("int32"), "source": Value("string"),
                            "embodiment": Value("string"), "bucket": Value("string"), "speed_mean": Value("float32"),
                            "heading_deg": Value("float32"), "dist_m": Value("float32"), "speed_first": Value("float32"),
                            "speed_last": Value("float32")})
SHARD_BYTES = 350e6


def shrink(blob, max_side, quality):
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="/dev/shm/dedup/selected")
    ap.add_argument("--out", default="/dev/shm/dedup/publish")
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_pretrain_dedup")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    sel = pd.read_parquet(f"{args.selected}/samples.parquet")
    out = Path(args.out)
    for sub in ("frames", "episodes", "samples"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if args.upload:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    all_eps, shard_no, pending_rows, pending_size, total = [], 0, [], 0, dict(frames=0, images=0, episodes=0)

    def flush(force=False):
        nonlocal pending_rows, pending_size, shard_no
        if pending_rows and (force or pending_size >= SHARD_BYTES):
            p = out / "frames" / f"train-{shard_no:05d}.parquet"
            fr.write_frames(pending_rows, p)
            if args.upload:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=f"data/frames/{p.name}", repo_id=args.repo,
                                repo_type="dataset", commit_message=f"frames shard {shard_no}")
            shard_no += 1
            pending_rows, pending_size = [], 0

    for repo, by_repo in sel.groupby("repo"):
        src = _Source(repo, "train")
        episodes = {e["episode_id"]: e for e in src.load_episodes()}
        for shard, by_shard in by_repo.groupby("shard"):
            ds = read_shard(repo, shard)
            tab = ds.data
            ep_col = np.asarray(tab.column("episode_id").to_pylist())
            fi_col = np.asarray(tab.column("frame_index").to_pylist())
            t_col = np.asarray(tab.column("timestamp").to_pylist())
            need = defaultdict(set)  # episode -> frame indices that need an image
            for r in by_shard.itertuples():
                e, fi = r.episode_id, int(r.frame_index)
                rows = np.flatnonzero(ep_col == e)
                t0 = t_col[rows][fi_col[rows] == fi][0]
                for dt in list(range(-5, 1)) + [5]:
                    j = rows[np.argmin(np.abs(t_col[rows] - (t0 + dt)))]
                    need[e].add(int(fi_col[j]))
            images = tab.column("image")
            for e, frames_needed in need.items():
                rows = np.flatnonzero(ep_col == e)
                order = rows[np.argsort(fi_col[rows])]
                with ThreadPoolExecutor(16) as pool:
                    blobs = dict(zip(sorted(frames_needed), pool.map(
                        lambda f: shrink(images[int(order[f])].as_py()["bytes"], args.max_side, args.quality), sorted(frames_needed))))
                for j in order:
                    fi = int(fi_col[j])
                    pending_rows.append(dict(episode_id=e, frame_index=fi, source_frame=int(tab.column("source_frame")[j].as_py()),
                                             timestamp=float(t_col[j]), image=({"bytes": blobs[fi], "path": None} if fi in blobs else None),
                                             image_right=None, pose=list(tab.column("pose")[j].as_py())))
                    pending_size += len(blobs[fi]) if fi in blobs else 0
                total["frames"] += len(order)
                total["images"] += len(blobs)
                total["episodes"] += 1
                ep = dict(episodes[e])
                all_eps.append(ep)
                flush()
            print(f"{repo.split('/')[1]} {Path(shard).stem}: {len(need)} episodes, {sum(len(v) for v in need.values())} images", flush=True)
    flush(force=True)
    fr.write_episodes(all_eps, out / "episodes" / "train-00000.parquet")
    samples = sel.assign(source=sel["repo"].str.split("/").str[1])[list(SAMPLE_FEATURES)]
    Dataset.from_pandas(samples.reset_index(drop=True), features=SAMPLE_FEATURES).to_parquet(out / "samples" / "train-00000.parquet")
    json.dump(dict(total, samples=int(len(sel)), shards=shard_no), open(out / "totals.json", "w"))
    print("totals", total, "samples", len(sel), "shards", shard_no)
    if args.upload:
        for sub in ("episodes", "samples"):
            api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(out / sub), path_in_repo=f"data/{sub}",
                              commit_message=f"{sub} table")


if __name__ == "__main__":
    main()
