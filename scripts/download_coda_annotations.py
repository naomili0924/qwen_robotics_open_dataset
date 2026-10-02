#!/usr/bin/env python
"""Download only the annotation-level parts of CODa (no images / point clouds).

The official archive (CODa_full_split.zip) is ~163 GB, but the parts needed to
build trajectory scenarios (3D boxes, poses, timestamps, calibrations, metadata)
are ~200 MB and sit in a few contiguous byte spans of the zip.  The server
supports HTTP range requests, so we read the zip central directory remotely and
fetch just those spans.

Usage:
    python scripts/download_coda_annotations.py --out data/raw/coda
"""
import argparse
import collections
import struct
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip
from tqdm import tqdm

URL = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_full_split.zip"
WANTED = ("3d_bbox", "3d_semantic", "timestamps", "metadata", "calibrations", "poses/dense", "poses/dense_global")
ROOT = "CODa_full/"
# Local file headers are 30 bytes + name + extra; pad the span end so the last
# member's data is fully covered whatever its header size.
LOCAL_HEADER_SLACK = 4096
MAX_GAP = 8 << 20  # merge members into one request while gaps stay below this


def group_of(name):
    rel = name[len(ROOT):]
    for w in WANTED:
        if rel.startswith(w + "/"):
            return w
    return None


def fetch_range(session, lo, hi):
    r = session.get(URL, headers={"Range": f"bytes={lo}-{hi - 1}"}, timeout=600)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"server ignored range request (status {r.status_code})")
    return r.content


def extract_member(buf, base, info):
    off = info.header_offset - base
    sig, _, _, method, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", buf[off:off + 30])
    if sig != 0x04034B50:
        raise RuntimeError(f"bad local header for {info.filename}")
    start = off + 30 + nlen + elen
    data = buf[start:start + info.compress_size]
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise RuntimeError(f"unsupported compression method {method} for {info.filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/coda")
    args = ap.parse_args()
    out = Path(args.out)

    with RemoteZip(URL) as z:
        infos = z.infolist()
    members = [i for i in infos if not i.is_dir() and group_of(i.filename)]
    members.sort(key=lambda i: i.header_offset)

    # Coalesce into contiguous spans.
    spans = []
    for i in members:
        end = i.header_offset + i.compress_size + LOCAL_HEADER_SLACK
        if spans and i.header_offset - spans[-1][1] < MAX_GAP:
            spans[-1][1] = max(spans[-1][1], end)
            spans[-1][2].append(i)
        else:
            spans.append([i.header_offset, end, [i]])

    total = sum(hi - lo for lo, hi, _ in spans)
    print(f"{len(members)} files in {len(spans)} spans, {total / 1e6:.1f} MB to fetch")
    counts = collections.Counter()
    session = requests.Session()
    for lo, hi, group in tqdm(spans, desc="spans"):
        buf = fetch_range(session, lo, hi)
        for info in group:
            data = extract_member(buf, lo, info)
            if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
                raise RuntimeError(f"size/CRC mismatch for {info.filename}")
            dst = out / info.filename[len(ROOT):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            counts[group_of(info.filename)] += 1
    for k, v in sorted(counts.items()):
        print(f"  {k:20s} {v} files")


if __name__ == "__main__":
    main()
