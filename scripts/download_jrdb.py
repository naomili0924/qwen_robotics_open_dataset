#!/usr/bin/env python
"""Download selected parts of the JRDB train archive over HTTP range requests.

The archive is one 75 GB zip; members of one kind sit in contiguous byte spans,
so each group is fetched as a few large ranged requests (in parallel chunks,
retried: the mirror answers roughly every other request with 503) and extracted
locally without storing the archive.

Usage:
    python scripts/download_jrdb.py --out data/raw/jrdb --groups labels,timestamps,calibration
    python scripts/download_jrdb.py --out /dev/shm/hnod/jrdb --groups images/image_0,pointclouds/upper_velodyne
"""
import argparse
import os
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))

URL = "http://download.cs.stanford.edu/downloads/jrdb/jrdb_train.zip"
ROOT = "train_dataset_with_activity/"
CHUNK = 16 << 20
LOCAL_HEADER_SLACK = 4096
MAX_GAP = 8 << 20


def session():
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=Retry(total=12, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504],
                                                     allowed_methods=["GET", "HEAD"]), pool_maxsize=32))
    return s


_local = threading.local()


def fetch(lo, hi, tries=40):
    """One ranged request, retried quickly: the mirror answers about every other request with 503."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    for attempt in range(tries):
        try:
            r = _local.s.get(URL, headers={"Range": f"bytes={lo}-{hi - 1}"}, timeout=(20, 120))
            if r.status_code == 206 and len(r.content) == hi - lo:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"failed to fetch bytes {lo}-{hi}")


def fetch_span(lo, hi, conns):
    """One contiguous byte span, in parallel chunks."""
    parts = [(a, min(a + CHUNK, hi)) for a in range(lo, hi, CHUNK)]
    with ThreadPoolExecutor(conns) as ex:
        return b"".join(ex.map(lambda p: fetch(*p), parts))


def extract(buf, base, m):
    off = m["offset"] - base
    sig, _, _, method, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", buf[off:off + 30])
    assert sig == 0x04034B50, m["name"]
    data = buf[off + 30 + nlen + elen:off + 30 + nlen + elen + m["csize"]]
    data = zlib.decompress(data, -15) if method == 8 else data
    assert len(data) == m["size"] and (zlib.crc32(data) & 0xFFFFFFFF) == m["crc"], m["name"]
    return data


def spans_of(members):
    members = sorted(members, key=lambda m: m["offset"])
    spans = []
    for m in members:
        end = m["offset"] + m["csize"] + LOCAL_HEADER_SLACK
        if spans and m["offset"] - spans[-1][1] < MAX_GAP:
            spans[-1][1] = max(spans[-1][1], end)
            spans[-1][2].append(m)
        else:
            spans.append([m["offset"], end, [m]])
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/jrdb")
    ap.add_argument("--groups", default="labels,timestamps,calibration",
                    help="comma-separated path prefixes under the archive root")
    ap.add_argument("--conns", type=int, default=12)
    ap.add_argument("--span-bytes", type=int, default=2 << 30, help="split larger spans into pieces of this size")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]  # directory prefixes or exact file paths

    with RemoteZip(URL, session=session()) as z:
        infos = z.infolist()
    members = [dict(name=i.filename, offset=i.header_offset, csize=i.compress_size, size=i.file_size, crc=i.CRC)
               for i in infos if not i.is_dir() and any(i.filename[len(ROOT):].startswith(g) for g in groups)]
    members = [m for m in members if not Path(args.out, m["name"][len(ROOT):]).exists()]
    spans = spans_of(members)
    # split big spans so memory stays bounded
    pieces = []
    for lo, hi, ms in spans:
        cur, start = [], None
        for m in ms:
            if cur and m["offset"] + m["csize"] + LOCAL_HEADER_SLACK - start > args.span_bytes:
                pieces.append((start, cur[-1]["offset"] + cur[-1]["csize"] + LOCAL_HEADER_SLACK, cur))
                cur = []
            if not cur:
                start = m["offset"]
            cur.append(m)
        if cur:
            pieces.append((start, cur[-1]["offset"] + cur[-1]["csize"] + LOCAL_HEADER_SLACK, cur))
    total = sum(hi - lo for lo, hi, _ in pieces)
    print(f"{len(members)} files in {len(pieces)} requests, {total / 1e9:.2f} GB", flush=True)
    for lo, hi, ms in tqdm(pieces, mininterval=10):
        buf = fetch_span(lo, hi, args.conns)
        for m in ms:
            dst = Path(args.out) / m["name"][len(ROOT):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(extract(buf, lo, m))


if __name__ == "__main__":
    main()
