#!/usr/bin/env python
"""Stream CODa front-camera images from the remote archive and re-encode them as JPEG.

The rectified cam0 PNGs of the labelled frames are ~59 GB inside the 163 GB
archive.  Each worker fetches one contiguous byte range of the zip, decodes the
PNGs in memory and writes <out>/<sequence>/<frame>.jpg (about 7 GB in total).
Re-running skips finished batches.

Usage:
    python scripts/stream_coda_images.py --raw data/raw/coda --out /dev/shm/hnod/coda_images
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from remotezip import RemoteZip
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_coda_lidar import LOCAL_HEADER_SLACK, URL, extract, fetch_range, make_batches  # noqa: E402

CAMERA = "cam0"
NAME_RE = re.compile(rf"2d_rect/{CAMERA}/(\d+)/2d_rect_{CAMERA}_\d+_(\d+)\.png$")
JPEG_QUALITY = 90


def run_batch(args):
    bid, members, out = args
    marker = Path(out) / f".batch_{bid:04d}.done"
    if marker.exists():
        return 0
    lo = members[0]["offset"]
    hi = max(m["offset"] + m["csize"] for m in members) + LOCAL_HEADER_SLACK
    buf = fetch_range(lo, hi)
    for m in members:
        img = cv2.imdecode(np.frombuffer(extract(buf, lo, m), np.uint8), cv2.IMREAD_COLOR)
        dst = Path(out) / str(m["seq"]) / f"{m['frame']}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    marker.touch()
    return len(members)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/coda")
    ap.add_argument("--out", default="/dev/shm/hnod/coda_images")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

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
    print(f"{len(members)} images in {len(batches)} batches", flush=True)
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_batch, (b, ms, args.out)) for b, ms in enumerate(batches)]
        for fut in tqdm(as_completed(futs), total=len(futs), mininterval=30):
            done += fut.result()
    print(f"processed {done} new images", flush=True)


if __name__ == "__main__":
    main()
