#!/usr/bin/env python
"""Stream RoboSense front-camera images from Hugging Face, undistort and re-encode them.

The image archive is one 170 GB tar.gz in 16 parts holding eight cameras; only the
front pinhole camera (images/0) of frames that end up in scenarios is kept.  Images
are undistorted with the calibration in the annotation files, so the published
intrinsics describe them without distortion terms.  Output: <out>/<source path>.

Needs `pigz` on PATH; see stream_robosense_lidar.py for how the archive is read.

Usage:
    python scripts/stream_robosense_images.py --pkl data/raw/robosense/splits --out data/interim/robosense_images
"""
import argparse
import multiprocessing
import os
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnod import robosense  # noqa: E402
from hnod.scenario import N_STEPS  # noqa: E402
from stream_robosense_lidar import REPO, feeder  # noqa: E402

PREFIX = "dataset/image_trainval_part_"
JPEG_QUALITY = 90


def undistort(args):
    raw, K, dist, dst = args
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.undistort(img, K, dist)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/raw/robosense/splits")
    ap.add_argument("--out", default="data/interim/robosense_images")
    ap.add_argument("--cache", default="/dev/shm/robosense_image_parts")
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    need = {}
    for split in ("train", "val"):
        for seg in robosense.load_segments(args.pkl, split, min_frames=N_STEPS):
            need.update(zip(seg["image_paths"], seg["image_calib"]))
    todo = {k for k in need if not os.path.exists(args.out + k)}
    print(f"{len(need)} images needed, {len(todo)} to do", flush=True)

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    parts = sorted((e for e in HfApi(token=token).list_repo_tree(REPO, "dataset", repo_type="dataset")
                    if e.path.startswith(PREFIX)), key=lambda e: e.path)
    print(f"{len(parts)} parts, {sum(p.size for p in parts) / 1e9:.0f} GB", flush=True)

    unzip = subprocess.Popen(["pigz", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    threading.Thread(target=feeder, args=(parts, args.cache, headers, unzip.stdin, args.lookahead, args.conns),
                     daemon=True).start()

    futures, n_seen, t0 = [], 0, time.time()
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool, \
            tarfile.open(fileobj=unzip.stdout, mode="r|", bufsize=16 << 20) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key = "/" + member.name.split("/", 1)[1]
            if key not in todo:
                continue
            K, dist = need[key]
            futures.append(pool.submit(undistort, (tar.extractfile(member).read(), K, dist, args.out + key)))
            n_seen += 1
            todo.discard(key)
            while len(futures) > 4 * args.workers:
                futures.pop(0).result()
            if n_seen % 2000 == 0:
                print(f"{n_seen} images, {time.time() - t0:.0f} s", flush=True)
        for fut in futures:
            fut.result()
    unzip.kill()
    print(f"processed {n_seen} new images in {time.time() - t0:.0f} s, {len(todo)} not found", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
