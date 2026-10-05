#!/usr/bin/env python
"""Convert EgoWalk (HF EgoWalk/trajectories, MIT) to the per-frame format and publish it.

Per recording: download the pose table, the RGB video and both language annotation sets, cut
episodes at odometry failures (nulls, re-initialisation jumps), re-encode every frame as JPEG at
native resolution, estimate indoor / outdoor per frame with CLIP, write
data/frames/<split>-<recording>.parquet and data/episodes/<split>-<recording>.parquet, upload them
and delete the local copies.  Depth video is not used (the model sees RGB only).  Resumable: a
recording whose episodes file is already on the Hub is skipped, so it can continue on any machine.

    python scripts/convert_egowalk.py --repo Jinyan0924/qwen_robotics_open_dataset_egowalk --workers 24
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import frames as fr  # noqa: E402
from hnod.environment import METHOD, IndoorTagger, per_frame, summarise  # noqa: E402

SRC = "EgoWalk/trajectories"
LICENCE = "MIT (EgoWalk)"
TAG_EVERY = 5          # frames between CLIP samples (1 s at 5 Hz)
TAG_WIDTH = 336


def _download(path, work, tries=8):
    for k in range(tries):
        try:
            return hf_hub_download(SRC, path, repo_type="dataset", local_dir=work)
        except Exception as e:  # 429s and network hiccups
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                return None
            time.sleep(min(300, 10 * 2 ** k))
    raise RuntimeError(f"could not download {path}")


def _segments(ann, a, b, source):
    out = []
    if ann is None:
        return out
    for r in ann.itertuples():
        if a <= r.start_frame and r.end_frame < b and r.end_frame > r.start_frame and str(r.caption).strip():
            out.append(dict(start_frame=int(r.start_frame - a), end_frame=int(r.end_frame - a), task="language",
                            instruction=str(r.caption).strip(), brief=str(r.brief or "").strip(), source=source))
    return out


def convert_one(name, work, out, heights, camera, quality):
    """Worker: write frames parquet; return episode rows (environment filled in later) and thumbnails."""
    rec = Path(work) / name
    data = pd.read_parquet(_download(f"data/{name}.parquet", rec)).sort_values("frame").reset_index(drop=True)
    video = _download(f"video/rgb/{name}__rgb.mp4", rec)
    anns = {}
    for src in ("end2end", "goal_boxes"):
        p = _download(f"annotations/{src}/{name}.parquet", rec)
        anns[src] = pd.read_parquet(p) if p else None
    assert (data["frame"].to_numpy() == np.arange(len(data))).all(), "frame numbers are not 0..n-1"

    t = data["timestamp"].to_numpy() / 1000.0
    xyz = data[["cart_x", "cart_y", "cart_z"]].to_numpy(float)
    valid = np.isfinite(xyz).all(1) & data["quat_w"].notna().to_numpy()
    yaw = fr.yaw_from_quat(*(data[c].to_numpy(float) for c in ("quat_x", "quat_y", "quat_z", "quat_w")))
    cuts = fr.cut_episodes(t, xyz[:, :2], valid)
    split = fr.split_of(name)

    which = np.full(len(data), -1)
    for e, (a, b) in enumerate(cuts):
        which[a:b] = e
    rows = {e: [] for e in range(len(cuts))}
    thumbs = {e: ([], []) for e in range(len(cuts))}
    cap = cv2.VideoCapture(video)
    shape = None
    for i in range(len(data)):
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"{name}: video ended at frame {i} of {len(data)}")
        shape = img.shape
        e = which[i]
        if e < 0:
            continue
        a = cuts[e][0]
        eid = f"egowalk/{name}/{e:02d}"
        rows[e].append(dict(episode_id=eid, frame_index=i - a, source_frame=i, timestamp=float(t[i] - t[a]),
                            image={"bytes": fr.jpeg_bytes(img, quality), "path": None}, image_right=None,
                            pose=[float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]), float(yaw[i])]))
        if (i - a) % TAG_EVERY == 0:
            h = int(round(img.shape[0] * TAG_WIDTH / img.shape[1]))
            thumbs[e][0].append(cv2.resize(img, (TAG_WIDTH, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy())
            thumbs[e][1].append(i - a)
    cap.release()

    episodes, all_rows = [], []
    for e, (a, b) in enumerate(cuts):
        length, speed = fr.path_stats(t[a:b], xyz[a:b, :2])
        tail = t[a:b] >= t[b - 1] - 2.0
        tail_speed = fr.path_stats(t[a:b][tail], xyz[a:b, :2][tail])[1] if tail.sum() > 1 else 1.0
        segs = _segments(anns["end2end"], a, b, "egowalk:end2end") + _segments(anns["goal_boxes"], a, b,
                                                                               "egowalk:goal_boxes")
        K = [camera["fx"], 0.0, camera["cx"], 0.0, camera["fy"], camera["cy"], 0.0, 0.0, 1.0]
        dist = [camera[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
        episodes.append(dict(
            episode_id=f"egowalk/{name}/{e:02d}", dataset="egowalk", source_sequence=name, split=split,
            num_frames=b - a, rate_hz=5.0, duration_s=float(t[b - 1] - t[a]), path_length_m=length,
            median_speed_mps=speed, embodiment="person_walking", environment="unknown", environment_method=METHOD,
            indoor_fraction=float("nan"), frame_indoor_prob=[],
            camera=dict(name="zed_left_rgb", width=int(shape[1]), height=int(shape[0]), K=K,
                        distortion_model="opencv_rational", distortion=dist,
                        height_m=float(heights.get(name, float("nan"))), stereo_baseline_m=float("nan")),
            segments=segs,
            # a recording that ends with the walker standing still is a real stop; cuts at odometry failures are not
            ends_at_rest=bool(b == len(data) and tail_speed < 0.15),
            fits="both" if segs else "robotnav", licence=LICENCE))
        all_rows += rows[e]
    out = Path(out)
    frames_path = out / "frames" / f"{split}-{name}.parquet"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        fr.write_frames(all_rows, frames_path)
    shutil.rmtree(rec, ignore_errors=True)
    return name, split, episodes, thumbs, (str(frames_path) if all_rows else None), len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", default="/dev/shm/egowalk/raw")
    ap.add_argument("--out", default="/dev/shm/egowalk/out")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="convert at most this many recordings")
    ap.add_argument("--only", nargs="*", help="recording names")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--commit-every", type=int, default=6)
    args = ap.parse_args()

    meta = Path(args.work) / "_meta"
    names = json.load(open(_download("meta/trajectories.json", meta)))
    heights = json.load(open(_download("meta/heights.json", meta)))
    camera = json.load(open(_download("meta/camera_rgb.json", meta)))
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    done = set()
    if not args.no_upload:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        done = {Path(f).stem.split("-", 1)[1] for f in api.list_repo_files(args.repo, repo_type="dataset")
                if f.startswith("data/episodes/")}
    todo = [n for n in (args.only or names) if n not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} recordings already on the Hub, {len(todo)} to convert", flush=True)

    tagger = IndoorTagger()
    pending, stats = [], dict(recordings=0, frames=0, episodes=0)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(convert_one, n, args.work, args.out, heights, camera, args.quality) for n in todo]
        for fut in as_completed(futs):
            try:
                name, split, episodes, thumbs, frames_path, n_src = fut.result()
            except Exception as e:  # one bad recording must not end the run; it stays to-do for the next run
                print("FAILED", repr(e)[:300], flush=True)
                continue
            for e, ep in enumerate(episodes):
                imgs, idx = thumbs[e]
                prob = per_frame(tagger(imgs), idx, ep["num_frames"]) if imgs else np.zeros(0, np.float32)
                ep["environment"], ep["indoor_fraction"] = summarise(prob)
                ep["frame_indoor_prob"] = prob.tolist()
            ep_path = Path(args.out) / "episodes" / f"{split}-{name}.parquet"
            ep_path.parent.mkdir(parents=True, exist_ok=True)
            fr.write_episodes(episodes, ep_path)
            n = sum(ep["num_frames"] for ep in episodes)
            stats["recordings"] += 1
            stats["frames"] += n
            stats["episodes"] += len(episodes)
            envs = ",".join(sorted({ep["environment"] for ep in episodes}))
            print(f"{name} {split}: {len(episodes)} episodes, {n}/{n_src} frames kept, {envs}", flush=True)
            if frames_path:
                pending.append((frames_path, f"data/frames/{Path(frames_path).name}"))
            pending.append((str(ep_path), f"data/episodes/{ep_path.name}"))  # last: marks the recording done
            if not args.no_upload and len(pending) >= 2 * args.commit_every:
                _commit(api, args.repo, pending)
                pending = []
    if pending and not args.no_upload:
        _commit(api, args.repo, pending)
    print("done", stats, flush=True)


def _commit(api, repo, pending):
    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=src) for src, dest in pending]
    for k in range(6):
        try:
            api.create_commit(repo, ops, repo_type="dataset", commit_message=f"Add {len(ops) // 2} EgoWalk recordings")
            break
        except Exception as e:
            print("commit failed, retrying:", repr(e)[:200], flush=True)
            time.sleep(30 * 2 ** k)
    else:
        raise RuntimeError("upload failed")
    for src, _ in pending:
        os.remove(src)


if __name__ == "__main__":
    main()
