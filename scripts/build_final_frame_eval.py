#!/usr/bin/env python
"""Add the final frame to the evaluation suite: config <version>_final_frame of the suite repo.

The final-frame task gives the model past frames and the one camera frame at the end of the scored horizon.
Suite rows hold only past and current images, so for every scenario the image at the end of the horizon is
taken from a later row of the same recording in the source repo (rows overlap in time).  Scenarios whose end
image exists nowhere (the recording stops) are left out and listed.

Horizon: the scenario's 10 future steps (5 s for CODa / MuSoHu / HSSD, 4 s for JRDB); RoboSense (1 Hz, 10 s)
uses its first 5 steps (5 s).  Adds columns final_image, final_step, horizon_s, embodiment, final_xy.

    python scripts/build_final_frame_eval.py --version v2 --upload
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, List, Value
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_suite import eval_features  # noqa: E402
from hnod.scenario import CURRENT, N_FUTURE  # noqa: E402

EMBODIMENT = {"musohu": "person_walking", "coda": "wheeled_robot", "jrdb": "wheeled_robot",
              "robosense": "wheeled_robot", "hssd": "simulated_agent"}
FINAL_STEP = {"robosense": 5}  # 1 Hz, 10 s horizon: use the first 5 s


def features():
    f = dict(eval_features())
    f.update(final_image=Image(), final_step=Value("int32"), horizon_s=Value("float32"),
             embodiment=Value("string"), final_xy=List(Value("float32")))
    return Features(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--suite", default="/workspace/cache/suite/suite/{version}/data")
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_eval")
    ap.add_argument("--work", default="/dev/shm/final_frame_eval")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = sorted(glob.glob(os.path.join(args.suite.format(version=args.version), "test-*.parquet")))
    if not files:  # no local copy: take the suite from the Hub
        files = [hf_hub_download(args.repo, f, repo_type="dataset", local_dir=args.work + "/suite")
                 for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith(f"data/{args.version}/test-")]
    rows = [r for f in files for r in pq.read_table(f).to_pylist()]

    # which source frame holds each scenario's final image
    want = defaultdict(dict)  # (repo, config, split) -> {(sequence, segment, source frame): suite_id}
    for r in rows:
        step = FINAL_STEP.get(r["source"], N_FUTURE)
        r["final_step"] = step
        key = (str(r["sequence"]), int(r["segment"]), int(r["source_frames"][CURRENT + step]))
        want[(r["source_repo"], r["source_config"], r["source_split"])][key] = r["suite_id"]

    found = {}
    for (repo, config, split), keys in want.items():
        shards = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(f"data/{config}/{split}-"))
        with ThreadPoolExecutor(8) as pool:
            local = list(pool.map(lambda f: hf_hub_download(repo, f, repo_type="dataset",
                                                            local_dir=f"{args.work}/src/{repo.split('/')[1]}"), shards))
        for path in local:
            pf = pq.ParquetFile(path)
            meta = pf.read(columns=["sequence", "segment", "source_frames"]).to_pylist()
            hits = {}  # row -> [(suite_id, image index)]
            for i, m in enumerate(meta):
                for k in range(CURRENT + 1):  # steps that have an image: the past and the current one
                    sid = keys.get((str(m["sequence"]), int(m["segment"]), int(m["source_frames"][k])))
                    if sid and sid not in found:
                        hits.setdefault(i, []).append((sid, k))
            if hits:
                images = pf.read(columns=["past_images"]).column("past_images")
                for i, lst in hits.items():
                    cell = images[i].as_py()
                    for sid, k in lst:
                        if cell and cell[k] is not None:
                            found[sid] = cell[k]["bytes"]
            os.remove(path)
        print(f"{repo.split('/')[1]} {config} {split}: {sum(s in found for s in keys.values())} of {len(keys)} final images",
              flush=True)

    out, missing = [], []
    for r in rows:
        if r["suite_id"] not in found:
            missing.append(r["suite_id"])
            continue
        step, e = r["final_step"], r["ego"]
        r.update(final_image={"bytes": found[r["suite_id"]], "path": None}, horizon_s=float(step / r["rate_hz"]),
                 embodiment=EMBODIMENT[r["source"]],
                 final_xy=[float(e["x"][CURRENT + step] - e["x"][CURRENT]), float(e["y"][CURRENT + step] - e["y"][CURRENT])])
        out.append(r)
    name = f"{args.version}_final_frame"
    dest = Path(args.work) / name
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("test-*.parquet"):
        old.unlink()
    per = 40
    n = (len(out) + per - 1) // per
    for k in range(n):
        Dataset.from_list(out[k * per:(k + 1) * per], features=features()).to_parquet(dest / f"test-{k:05d}-of-{n:05d}.parquet")
    json.dump(dict(kept=len(out), missing=missing), open(dest / "missing.json", "w"), indent=1)
    by = defaultdict(int)
    for r in out:
        by[(r["environment"], r["source"])] += 1
    print(f"{name}: {len(out)} scenarios kept, {len(missing)} without a final image: {missing}")
    print(dict(by))
    if args.upload:
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(dest), path_in_repo=f"data/{name}",
                          commit_message=f"{name}: suite {args.version} with the frame at the end of the horizon")


if __name__ == "__main__":
    main()
