#!/usr/bin/env python
"""Pass 2 of the deduplication: choose a motion-diverse, then scene-diverse subset of the indexed samples.

Per source and motion class (build_dedup_index.bucket):
  1. rare classes (turns, stops, starts, slowing, speeding up, weaving) are kept in full up to --cap-rare;
  2. "straight" and "standing" are capped at --cap-straight / --cap-standing;
  3. inside a class, samples are taken in random order and a sample is skipped when its scene embedding is more
     than --sim like one already kept in that class (same place, same view, same motion = duplicate);
  4. simulated houses: at most --per-house samples per house.
Writes <out>/samples.parquet (one row per kept sample) and <out>/summary.json.

    python scripts/select_dedup.py --index /dev/shm/dedup/index --out /dev/shm/dedup/selected
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RARE = ("sharp_left", "sharp_right", "turn_left", "turn_right", "stopping", "starting", "slowing", "speeding_up", "weaving")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/dev/shm/dedup/index")
    ap.add_argument("--out", default="/dev/shm/dedup/selected")
    ap.add_argument("--sim", type=float, default=0.90)
    ap.add_argument("--sim-sim", type=float, default=0.96, help="threshold for simulated renders (HSSD), which all look alike")
    ap.add_argument("--cap-rare", type=int, default=6000, help="per source and rare class")
    ap.add_argument("--cap-straight", type=int, default=6000, help="per source")
    ap.add_argument("--cap-standing", type=int, default=1500, help="per source")
    ap.add_argument("--per-house", type=int, default=120, help="simulated houses (HSSD)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{args.index}/*.parquet"))], ignore_index=True)
    E = np.stack([np.frombuffer(b, np.float16) for b in df["embedding"]]).astype(np.float32)
    df = df.drop(columns=["embedding"])
    df["house"] = np.where(df["repo"].str.contains("hssd"), df["episode_id"].str.split("/").str[1].str.split("_").str[0], "")
    rng = np.random.default_rng(args.seed)
    keep = np.zeros(len(df), bool)
    before = Counter(zip(df["repo"], df["bucket"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (repo, b), idx in df.groupby(["repo", "bucket"]).indices.items():
        cap = args.cap_standing if b == "standing" else args.cap_straight if b == "straight" else args.cap_rare
        order = idx[rng.permutation(len(idx))]
        kept_emb = torch.zeros((0, E.shape[1]), device=dev)
        per_house = Counter()
        n = 0
        for i in order:
            if n >= cap:
                break
            h = df["house"].iat[i]
            if h and per_house[h] >= args.per_house:
                continue
            e = torch.from_numpy(E[i]).to(dev)
            thr = args.sim_sim if "hssd" in repo else args.sim
            if len(kept_emb) and float((kept_emb @ e).max()) > thr:
                continue
            kept_emb = torch.cat([kept_emb, e[None]])
            keep[i] = True
            per_house[h] += 1
            n += 1
    sel = df[keep].copy()
    after = Counter(zip(sel["repo"], sel["bucket"]))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sel.to_parquet(f"{args.out}/samples.parquet")
    summary = dict(before=int(len(df)), after=int(len(sel)), sim=args.sim,
                   per_source={r.split("/")[1]: dict(before=int((df["repo"] == r).sum()), after=int((sel["repo"] == r).sum()))
                               for r in df["repo"].unique()},
                   buckets={r.split("/")[1]: {b: dict(before=before[(r, b)], after=after[(r, b)]) for b in sorted(df["bucket"].unique())}
                            for r in df["repo"].unique()})
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    json.dump(dict(vars(args), max_side=640), open(f"{args.out}/args.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "buckets"}, indent=1))
    for r, bs in summary["buckets"].items():
        print(r, {b: f"{v['before']}->{v['after']}" for b, v in bs.items()})
    # embeddings of the kept samples, for the card's figures
    np.save(f"{args.out}/embeddings_kept.npy", E[keep].astype(np.float16))
    np.save(f"{args.out}/embeddings_sample_before.npy", E[rng.choice(len(E), min(20000, len(E)), replace=False)].astype(np.float16))


if __name__ == "__main__":
    main()
