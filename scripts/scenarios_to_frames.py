#!/usr/bin/env python
"""Rebuild per-frame training data (hnod.frames) from a published scenario config.

Scenario rows (hnod.io) carry the current and 10 past images with the robot's poses, and consecutive
rows overlap, so every imaged frame of a recording appears at least once.  This collects each frame
once, maps its pose back to the source's world frame (``world_from_scenario``), cuts episodes at
gaps, estimates indoor / outdoor per frame unless the environment is known, and publishes the
``frames`` and ``episodes`` configs (same splits as the source).  The last second or so of each
recorded segment has poses but no images in the scenario rows and is dropped.

    python scripts/scenarios_to_frames.py --src Jinyan0924/qwen_robotics_open_dataset_jrdb --config jrdb_5hz \
        --dest Jinyan0924/qwen_robotics_open_dataset_jrdb --dataset jrdb --embodiment wheeled_robot \
        --licence "CC BY-NC-SA 3.0 (JRDB)"
"""
import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import frames as fr  # noqa: E402
from hnod.environment import METHOD, IndoorTagger, per_frame, summarise  # noqa: E402

COLUMNS = ["sequence", "segment", "rate_hz", "current_time_index", "timestamps", "source_frames",
           "world_from_scenario", "past_images", "camera", "ego"]
SHARD_BYTES = 400e6


def collect(paths):
    """{(sequence, segment): {source_frame: (t, pose, image bytes)}} and one camera per key."""
    out, cams, rates = defaultdict(dict), {}, {}
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=64, columns=COLUMNS):
            for r in batch.to_pylist():
                key = (str(r["sequence"]), int(r["segment"]))
                T = np.asarray(r["world_from_scenario"], float).reshape(4, 4)
                yaw_T = np.arctan2(T[1, 0], T[0, 0])
                ego, now = r["ego"], r["current_time_index"]
                cams.setdefault(key, r["camera"])
                rates[key] = r["rate_hz"]
                for k in range(now + 1):
                    f = int(r["source_frames"][k])
                    if f in out[key] or not r["past_images"] or r["past_images"][k] is None:
                        continue
                    w = T @ np.array([ego["x"][k], ego["y"][k], 0.0, 1.0])
                    yaw = float(np.arctan2(np.sin(ego["heading"][k] + yaw_T), np.cos(ego["heading"][k] + yaw_T)))
                    out[key][f] = (float(r["timestamps"][k]), [float(w[0]), float(w[1]), float(w[2]), yaw],
                                   r["past_images"][k]["bytes"])
    return out, cams, rates


def episodes_of(frames_by_no, rate_hz):
    """Cut a recording's frames into runs without gaps; returns lists of frame numbers."""
    nos = np.array(sorted(frames_by_no))
    t = np.array([frames_by_no[n][0] for n in nos])
    xy = np.array([frames_by_no[n][1][:2] for n in nos])
    cuts = fr.cut_episodes(t, xy, np.ones(len(nos), bool), max_step=3.0, max_gap_s=2.5 / rate_hz,
                           min_frames=8, min_length_m=1.0)
    return [nos[a:b] for a, b in cuts]


def camera_row(cam):
    # one flattened 4x4 camera pose per past image, scenario frame (origin on the ground); the last is now
    T = np.asarray(cam["T_scenario_from_camera"][-1], float).reshape(4, 4) if cam.get("T_scenario_from_camera") else None
    return dict(name=cam["name"], width=int(cam["width"]), height=int(cam["height"]), K=list(cam["K"]),
                distortion_model="none", distortion=[],
                height_m=float(T[2, 3]) if T is not None else float("nan"), stereo_baseline_m=float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embodiment", required=True, choices=sorted(fr.EMBODIMENTS))
    ap.add_argument("--environment", default="estimate", help="indoor | outdoor | estimate (CLIP)")
    ap.add_argument("--licence", required=True)
    ap.add_argument("--ends-at-rest", action="store_true", help="episodes end with the agent stopped (simulator)")
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--work", default="/dev/shm/s2f")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(args.src, repo_type="dataset")
    tagger = IndoorTagger() if args.environment == "estimate" else None
    out = Path(args.work) / args.dataset
    for split in args.splits:
        shards = sorted(f for f in files if f.startswith(f"data/{args.config}/{split}-"))
        if not shards:
            continue
        paths = [hf_hub_download(args.src, f, repo_type="dataset", local_dir=Path(args.work) / "src") for f in shards]
        by_key, cams, rates = collect(paths)
        for p in paths:
            os.remove(p)
        frame_rows, episode_rows = [], []
        for key in sorted(by_key):
            fs, rate = by_key[key], rates[key]
            for part, nos in enumerate(episodes_of(fs, rate)):
                eid = f"{args.dataset}/{key[0]}/{key[1]}" + (f"/{part}" if part else "")
                t0 = fs[nos[0]][0]
                rows = [dict(episode_id=eid, frame_index=j, source_frame=int(n), timestamp=fs[n][0] - t0,
                             image={"bytes": fs[n][2], "path": None}, image_right=None, pose=fs[n][1])
                        for j, n in enumerate(nos)]
                t = np.array([fs[n][0] for n in nos])
                xy = np.array([fs[n][1][:2] for n in nos])
                length, speed = fr.path_stats(t, xy)
                if tagger is not None:
                    every = max(1, int(round(rate)))
                    idx = list(range(0, len(nos), every))
                    imgs = [cv2.cvtColor(cv2.imdecode(np.frombuffer(fs[nos[j]][2], np.uint8), cv2.IMREAD_COLOR),
                                         cv2.COLOR_BGR2RGB) for j in idx]
                    prob = per_frame(tagger(imgs), idx, len(nos))
                    env, frac = summarise(prob)
                    method = METHOD
                else:
                    prob = np.full(len(nos), 1.0 if args.environment == "indoor" else 0.0, np.float32)
                    env, frac, method = args.environment, float(prob.mean()), "labelled"
                episode_rows.append(dict(
                    episode_id=eid, dataset=args.dataset, source_sequence=key[0], split=split, num_frames=len(nos),
                    rate_hz=float(rate), duration_s=float(t[-1] - t[0]), path_length_m=length,
                    median_speed_mps=speed, embodiment=args.embodiment, environment=env, environment_method=method,
                    indoor_fraction=frac, frame_indoor_prob=prob.tolist(), camera=camera_row(cams[key]),
                    segments=[], ends_at_rest=bool(args.ends_at_rest), fits="robotnav", licence=args.licence))
                frame_rows.append(rows)
        del by_key
        # shards of whole episodes, about SHARD_BYTES each
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / "episodes").mkdir(parents=True, exist_ok=True)
        groups, cur, size = [], [], 0
        for rows in frame_rows:
            cur += rows
            size += sum(len(r["image"]["bytes"]) for r in rows)
            if size >= SHARD_BYTES:
                groups.append(cur)
                cur, size = [], 0
        if cur:
            groups.append(cur)
        written = []
        for g, rows in enumerate(groups):
            p = out / "frames" / f"{split}-{g:05d}-of-{len(groups):05d}.parquet"
            fr.write_frames(rows, p)
            written.append(p)
        ep_path = out / "episodes" / f"{split}-00000-of-00001.parquet"
        fr.write_episodes(episode_rows, ep_path)
        n = sum(len(r) for r in frame_rows)
        envs = defaultdict(int)
        for e in episode_rows:
            envs[e["environment"]] += 1
        print(f"{split}: {len(episode_rows)} episodes, {n} frames, {len(groups)} shards, {dict(envs)}", flush=True)
        if not args.no_upload:
            api.create_repo(args.dest, repo_type="dataset", exist_ok=True)
            for sub in ("frames", "episodes"):
                api.upload_folder(repo_id=args.dest, repo_type="dataset", folder_path=str(out / sub),
                                  path_in_repo=f"data/{sub}", allow_patterns=[f"{split}-*.parquet"],
                                  delete_patterns=[f"{split}-*.parquet"],
                                  commit_message=f"Per-frame {args.dataset} {split} ({sub})")
            for p in written + [ep_path]:
                os.remove(p)


if __name__ == "__main__":
    tempfile.tempdir = "/dev/shm"
    main()
