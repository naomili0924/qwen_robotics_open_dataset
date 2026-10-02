#!/usr/bin/env python
"""Stream RoboSense Hesai sweeps from Hugging Face and reduce each to a BEV grid.

The lidar/occupancy archive is one 244 GB tar.gz cut into 23 parts, so it can
only be read front to back.  Parts are fetched ahead of time into a RAM disk
with many parallel range requests, fed in order through `pigz -dc`, and the
tar stream is walked once; only the Hesai sweeps of labelled frames that end up
in scenarios are decoded (see hnod/lidar_bev.py), everything else is skipped.
Per sweep a tri-state grid (batch_*.pkl) and a thinned point cloud with a ground
flag (points_*.pkl) are kept, a few GB in total.

Needs `pigz` on PATH and HF_TOKEN in the environment (optional for public repos,
but it lifts rate limits).

Usage:
    python scripts/stream_robosense_lidar.py --pkl data/raw/robosense/splits --out data/interim/robosense_bev
"""
import argparse
import multiprocessing
import os
import pickle
import subprocess
import sys
import tarfile
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import lidar_bev, robosense  # noqa: E402
from hnod.scenario import N_STEPS  # noqa: E402

REPO = "suhaisheng0527/RoboSense"
PREFIX = "dataset/lidar_occ_trainval_part_"
CHUNK = 128 << 20
BATCH = 2000


def fetch_part(url, size, dst, headers, conns):
    """Download one archive part to `dst` with parallel range requests."""
    fd = os.open(dst + ".tmp", os.O_CREAT | os.O_WRONLY)
    os.ftruncate(fd, size)

    def piece(lo):
        hi = min(lo + CHUNK, size)
        for attempt in range(8):
            try:
                r = requests.get(url, headers={**headers, "Range": f"bytes={lo}-{hi - 1}"}, timeout=300)
                if r.status_code == 206 and len(r.content) == hi - lo:
                    os.pwrite(fd, r.content, lo)
                    return
            except requests.RequestException:
                pass
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url} bytes {lo}-{hi}")

    with ThreadPoolExecutor(conns) as ex:
        list(ex.map(piece, range(0, size, CHUNK)))
    os.close(fd)
    os.rename(dst + ".tmp", dst)


def feeder(parts, cache, headers, sink, lookahead, conns):
    """Keep `lookahead` parts downloading and pour them, in order, into `sink`."""
    slots = threading.Semaphore(lookahead)
    with ThreadPoolExecutor(lookahead) as ex:
        def submit(part):
            slots.acquire()
            dst = os.path.join(cache, os.path.basename(part.path))
            url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{part.path}"
            return dst, ex.submit(fetch_part, url, part.size, dst, headers, conns)

        pending = []
        it = iter(parts)
        try:
            while True:
                while len(pending) < lookahead:
                    part = next(it, None)
                    if part is None:
                        break
                    pending.append(submit(part))
                if not pending:
                    break
                dst, fut = pending.pop(0)
                fut.result()
                with open(dst, "rb") as f:
                    while chunk := f.read(16 << 20):
                        sink.write(chunk)
                os.remove(dst)
                slots.release()
        except BrokenPipeError:
            pass  # reader stopped early (--max-frames)
        finally:
            try:
                sink.close()
            except BrokenPipeError:
                pass


def reduce_sweep(args):
    key, raw, (T, height) = args
    pts = np.frombuffer(raw, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    idx, fi, fj, h, origin = lidar_bev.point_heights(pts, T[:3, :3], T[:3, 3], height)
    packed = lidar_bev.pack_points(*lidar_bev.downsample_points(pts[idx], h))
    return key, (origin, zlib.compress(lidar_bev.rasterize(fi, fj, h).tobytes(), 6)), packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/raw/robosense/splits")
    ap.add_argument("--out", default="data/interim/robosense_bev")
    ap.add_argument("--cache", default="/dev/shm/robosense_parts")
    ap.add_argument("--lookahead", type=int, default=4, help="archive parts held in the RAM disk (10.7 GB each)")
    ap.add_argument("--conns", type=int, default=16, help="parallel connections per part")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-frames", type=int, help="stop after this many sweeps (for a pilot run)")
    ap.add_argument("--absent", help="text file of sweep paths known to be missing from the archive; lets a "
                                     "resumed run stop as soon as everything else is done")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    # Sweeps worth decoding: frames of chains long enough to yield a scenario.
    need = {}
    for split in ("train", "val"):
        for seg in robosense.load_segments(args.pkl, split, min_frames=N_STEPS):
            need.update(zip(seg["lidar_paths"], zip(seg["lidar_T"], seg["lidar_heights"])))
    done = set()
    for f in out.glob("batch_*.pkl"):
        done.update(pickle.load(open(f, "rb")))
    absent = set(Path(args.absent).read_text().split()) if args.absent and Path(args.absent).exists() else set()
    todo = set(need) - done - absent
    print(f"{len(need)} sweeps needed, {len(done)} already done, {len(todo)} to do", flush=True)

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    parts = sorted((e for e in HfApi(token=token).list_repo_tree(REPO, "dataset", repo_type="dataset")
                    if e.path.startswith(PREFIX)), key=lambda e: e.path)
    print(f"{len(parts)} parts, {sum(p.size for p in parts) / 1e9:.0f} GB", flush=True)

    unzip = subprocess.Popen(["pigz", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    feed = threading.Thread(target=feeder, args=(parts, args.cache, headers, unzip.stdin, args.lookahead, args.conns),
                            daemon=True)
    feed.start()

    results, points, futures, n_batch, n_seen, t0 = {}, {}, [], len(list(out.glob("batch_*.pkl"))), 0, time.time()

    def collect(fut):
        key, grid, packed = fut.result()
        results[key], points[key] = grid, packed

    def flush(force=False):
        nonlocal results, points, n_batch
        if results and (force or len(results) >= BATCH):
            with open(out / f"points_{n_batch:04d}.pkl", "wb") as f:
                pickle.dump(points, f)
            with open(out / f"batch_{n_batch:04d}.pkl", "wb") as f:
                pickle.dump(results, f)
            n_batch += 1
            results, points = {}, {}

    # forkserver: forked workers would inherit pigz's stdin and keep it from ever seeing end-of-file
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool, tarfile.open(fileobj=unzip.stdout, mode="r|", bufsize=16 << 20) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key = "/" + member.name.split("/", 1)[1]
            if key not in need or key in done:
                continue
            futures.append(pool.submit(reduce_sweep, (key, tar.extractfile(member).read(), need[key])))
            n_seen += 1
            todo.discard(key)
            while len(futures) > 4 * args.workers or (futures and futures[0].done()):
                collect(futures.pop(0))
                flush()
            if n_seen % 1000 == 0:
                print(f"{n_seen} sweeps, {time.time() - t0:.0f} s", flush=True)
            if (args.max_frames and n_seen >= args.max_frames) or (absent and not todo):
                break
        for fut in futures:
            collect(fut)
        flush(force=True)
    unzip.kill()
    print(f"processed {n_seen} new sweeps in {time.time() - t0:.0f} s", flush=True)
    os._exit(0)  # the feeder may still be blocked on a download when stopping early


if __name__ == "__main__":
    main()
