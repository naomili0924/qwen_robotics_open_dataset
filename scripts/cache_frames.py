#!/usr/bin/env python
"""Build a local training copy of per-frame repos with smaller images.

The policy resizes every image to at most cfg.max_pixels (448 x 448 by default), so full-resolution frames
only cost disk and decoding time.  This downloads each `frames` shard of a split, re-encodes the images so
the long side is at most --max-side pixels, writes <out>/<repo name>/frames/<split>-*.parquet (and copies the
`episodes` files), and deletes the download.  Train with --frames-repos <out>/<repo name>,...

    python scripts/cache_frames.py --repos Jinyan0924/qwen_robotics_open_dataset_egowalk ... --split train
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _shrink(blob, max_side, quality):
    if blob is None:
        return None
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def shrink_shard(job):
    repo, path, out, max_side, quality, work = job
    dest = Path(out) / path.split("/", 1)[1]  # frames/<split>-....parquet
    if dest.exists():
        return str(dest), 0
    local = hf_hub_download(repo, path, repo_type="dataset", local_dir=work)
    pf = pq.ParquetFile(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    writer, n = None, 0
    for batch in pf.iter_batches(batch_size=512):
        t = pa.Table.from_batches([batch])
        for col in ("image", "image_right"):
            if col in t.column_names:
                cells = t.column(col).to_pylist()
                new = [None if c is None or c.get("bytes") is None else
                       {"bytes": _shrink(c["bytes"], max_side, quality), "path": None} for c in cells]
                t = t.set_column(t.column_names.index(col), col, pa.array(new, type=t.schema.field(col).type))
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema)  # keeps the Hugging Face feature metadata
        writer.write_table(t)
        n += len(t)
    writer.close()
    tmp.rename(dest)
    os.remove(local)
    return str(dest), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="/dev/shm/train_cache")
    ap.add_argument("--work", default="/dev/shm/train_cache/_download")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    jobs = []
    for repo in args.repos:
        out = Path(args.out) / repo.split("/")[1]
        files = api.list_repo_files(repo, repo_type="dataset")
        eps = [f for f in sorted(files) if f.startswith(f"data/episodes/{args.split}-")
               and not (out / f.split("/", 1)[1]).exists()]
        if eps:  # small files: fetch them in parallel
            snapshot_download(repo, repo_type="dataset", allow_patterns=eps, local_dir=args.work, max_workers=16)
            for f in eps:
                dest = out / f.split("/", 1)[1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(Path(args.work) / f, dest)
        jobs += [(repo, f, str(out), args.max_side, args.quality, args.work) for f in sorted(files)
                 if f.startswith(f"data/frames/{args.split}-")]
    print(f"{len(jobs)} frame shards", flush=True)
    total = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (dest, n) in enumerate(pool.map(shrink_shard, jobs), 1):
            total += n
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} shards, {total} frames", flush=True)


if __name__ == "__main__":
    main()
