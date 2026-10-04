#!/usr/bin/env python
"""Upload a generator's finished scenes to the Hugging Face Hub while it is still running.

generate_habitat_pointnav.py keeps per-scene shards under <out>/<config>/_parts/<split>/ until
the whole run ends; on a rented machine that output dies with the instance.  This uploads every
scene that has its .done marker as data/<config>/<split>-<job>-NNNNN-of-NNNNN.parquet (the
dataset card globs <split>-*.parquet, so partial uploads are loadable), remembers what it sent,
and can loop.  Needs HF_TOKEN.

    python scripts/sync_parts_hf.py --repo <user>/<name> --out /dev/shm/hnod/hf_hssd --config hssd_2hz --every 900
"""
import argparse
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def sync(api, repo, parts, config, batch=40):
    have = set(api.list_repo_files(repo, repo_type="dataset"))
    ops = []
    for split in ("train", "validation", "test"):
        for done in sorted((parts / split).glob("*.done")):
            for f in sorted((parts / split).glob(f"{done.stem}-*.parquet")):
                dest = f"data/{config}/{split}-{f.name}"
                if dest not in have:
                    ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(f)))
    for i in range(0, len(ops), batch):
        api.create_commit(repo, ops[i:i + batch], repo_type="dataset",
                          commit_message=f"Add {len(ops[i:i + batch])} {config} shards")
    return len(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True, help="the generator's --out directory")
    ap.add_argument("--config", required=True)
    ap.add_argument("--every", type=float, default=0, help="repeat every N seconds (0: once)")
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    parts = Path(args.out) / args.config / "_parts"
    while True:
        try:
            n = sync(api, args.repo, parts, args.config)
            print(time.strftime("%H:%M:%S"), f"uploaded {n} new shards", flush=True)
        except Exception as e:  # network hiccups must not end the loop
            print(time.strftime("%H:%M:%S"), "sync failed:", repr(e)[:200], flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
