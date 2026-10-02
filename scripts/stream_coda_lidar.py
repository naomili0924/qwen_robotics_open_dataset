#!/usr/bin/env python
"""Stream CODa lidar sweeps from the remote archive and reduce each to a BEV grid.

The ego-motion-compensated sweeps (3d_comp, ~40 GB compressed) are never stored:
each worker fetches one contiguous byte range of the zip, decodes the sweeps in
memory and writes, per frame, a compressed tri-state grid (batch_*.pkl) and a
voxel-downsampled point cloud with a ground flag (points_*.pkl); see
hnod/lidar_bev.py.  Output is a few GB.  Re-running skips finished batches.

Usage:
    python scripts/stream_coda_lidar.py --raw data/raw/coda --out data/interim/coda_bev
"""
import argparse
import pickle
import re
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from remotezip import RemoteZip
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import lidar_bev  # noqa: E402
from hnod.coda import LIDAR_HEIGHT, load_poses  # noqa: E402

URL = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_full_split.zip"
NAME_RE = re.compile(r"3d_comp/os1/(\d+)/3d_comp_os1_\d+_(\d+)\.bin$")
BATCH_BYTES = 200 << 20
LOCAL_HEADER_SLACK = 4096


def fetch_range(lo, hi, tries=6):
    for attempt in range(tries):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={lo}-{hi - 1}"}, timeout=900)
            if r.status_code == 206 and len(r.content) == hi - lo:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch bytes {lo}-{hi}")


def extract(buf, base, m):
    off = m["offset"] - base
    sig, _, _, method, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", buf[off:off + 30])
    assert sig == 0x04034B50, m["name"]
    data = buf[off + 30 + nlen + elen:off + 30 + nlen + elen + m["csize"]]
    return zlib.decompress(data, -15) if method == 8 else data


def make_batches(members):
    """Split archive members into runs of at most BATCH_BYTES of contiguous archive bytes."""
    members = sorted(members, key=lambda m: m["offset"])
    batches, cur, start = [], [], None
    for m in members:
        if cur and (m["offset"] + m["csize"] - start > BATCH_BYTES):
            batches.append(cur)
            cur = []
        if not cur:
            start = m["offset"]
        cur.append(m)
    if cur:
        batches.append(cur)
    return batches


def semantic_stats(raw, seq, frame, idx, h):
    """Per terrain class: how many labelled points we call ground / obstacle.

    CODa's 3d_semantic labels exist for a subset of frames; class 0 is unlabelled
    (everything that is not terrain).  Returns (seq, frame, counts[25, 3]) with
    columns total / classified ground / classified obstacle.
    """
    f = Path(raw) / f"3d_semantic/os1/{seq}/3d_semantic_os1_{seq}_{frame}.bin"
    if not f.exists():
        return None
    lab = np.minimum(np.fromfile(f, dtype=np.uint8)[idx], 24)
    obst = (h > lidar_bev.OBST_MIN_H) & (h < lidar_bev.OBST_MAX_H)
    ground = h < lidar_bev.GROUND_MAX_H
    counts = np.stack([np.bincount(lab, minlength=25), np.bincount(lab[ground], minlength=25),
                       np.bincount(lab[obst], minlength=25)], axis=1)
    return seq, frame, counts


def run_batch(args):
    bid, members, raw, out = args
    dst = Path(out) / f"batch_{bid:04d}.pkl"
    if dst.exists():
        return bid, 0
    lo = members[0]["offset"]
    hi = max(m["offset"] + m["csize"] for m in members) + LOCAL_HEADER_SLACK
    buf = fetch_range(lo, hi)
    poses, frames, stats, points = {}, [], [], {}
    for m in members:
        seq, frame = m["seq"], m["frame"]
        if seq not in poses:
            poses[seq] = load_poses(raw, seq)
        pts = np.frombuffer(extract(buf, lo, m), dtype=np.float32).reshape(-1, 4)
        T = poses[seq][frame]
        idx, fi, fj, h, origin = lidar_bev.point_heights(pts, T[:3, :3], T[:3, 3], LIDAR_HEIGHT)
        grid = lidar_bev.rasterize(fi, fj, h)
        frames.append((seq, frame, origin, zlib.compress(grid.tobytes(), 6)))
        points[(seq, frame)] = lidar_bev.pack_points(*lidar_bev.downsample_points(pts[idx, :3], h))
        s = semantic_stats(raw, seq, frame, idx, h)
        if s:
            stats.append(s)
    with open(Path(out) / f"points_{bid:04d}.pkl", "wb") as f:
        pickle.dump(points, f)
    tmp = dst.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dict(frames=frames, stats=stats), f)
    tmp.rename(dst)
    return bid, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/coda")
    ap.add_argument("--out", default="data/interim/coda_bev")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    annotated = {p.name for p in (Path(args.raw) / "3d_bbox/os1").iterdir()}
    with RemoteZip(URL) as z:
        infos = z.infolist()
    members = []
    for i in infos:
        mt = NAME_RE.search(i.filename)
        if mt and mt.group(1) in annotated:
            members.append(dict(name=i.filename, offset=i.header_offset, csize=i.compress_size,
                                seq=int(mt.group(1)), frame=int(mt.group(2))))
    batches = make_batches(members)
    print(f"{len(members)} sweeps in {len(batches)} batches")

    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_batch, (b, ms, args.raw, str(out))) for b, ms in enumerate(batches)]
        for fut in tqdm(as_completed(futs), total=len(futs), mininterval=30):
            done += fut.result()[1]
    print(f"processed {done} new sweeps")


if __name__ == "__main__":
    main()
