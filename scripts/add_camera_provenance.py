#!/usr/bin/env python
"""One-off: record where each episode's camera height comes from and put the camera sentence into the published tables.

* episodes tables (training sources and the dedup repo): camera.height_source = dataset | simulator | unknown
  (EgoWalk's heights come from its meta/heights.json; CODa / RoboSense from their calibration; HSSD is rendered;
  a NaN height is unknown).  Nothing is estimated.
* dedup repo samples table: camera_prompt per sample (hnod.windows.camera_prompt of its episode's camera).
* final-frame eval configs: camera_height_m / camera_height_source / camera_prompt (MuSoHu: unknown).

    python scripts/add_camera_provenance.py --upload
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_frame_eval import camera_calibration, features  # noqa: E402
from hnod.frames import EPISODE_FEATURES  # noqa: E402
from hnod.windows import camera_prompt  # noqa: E402

SOURCE_OF = {"egowalk": "dataset", "coda": "dataset", "robosense": "dataset", "hssd": "simulator", "jrdb": "dataset"}
EPISODE_REPOS = ["Jinyan0924/qwen_robotics_nav_pretrain_dedup", "Jinyan0924/qwen_robotics_open_dataset_egowalk",
                 "Jinyan0924/qwen_robotics_open_dataset", "Jinyan0924/qwen_robotics_open_dataset_robosense",
                 "Jinyan0924/habitat_hssd_pointgoal_nav_scenarios"]


def with_source(cam, dataset):
    cam = dict(cam)
    hm = cam.get("height_m")
    known = hm is not None and np.isfinite(hm) and SOURCE_OF.get(dataset) is not None
    cam["height_source"] = SOURCE_OF[dataset] if known else "unknown"
    if not known:
        cam["height_m"] = float("nan")
    return cam


def upload(api, kw):
    """upload_folder with a wait on the Hub's 429 (commit rate limit)."""
    for attempt in range(6):
        try:
            return api.upload_folder(**kw)
        except Exception as e:  # noqa: BLE001
            if "429" not in str(e) or attempt == 5:
                raise
            print(f"rate limited, waiting {5 * 2 ** attempt} min", flush=True)
            time.sleep(300 * 2 ** attempt)


def export_calibration(cameras, work, out=Path(__file__).resolve().parents[1] / "docs" / "calibration"):
    """docs/calibration/*.jsonl: the calibration and its provenance per episode and per eval scenario (kept in git)."""
    import json
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes_camera.jsonl", "w") as f:
        for repo_dir in sorted(work.glob("*/data/episodes")):
            repo = "Jinyan0924/" + repo_dir.parts[-3]
            for fp in sorted(repo_dir.glob("*.parquet")):
                for r in pq.read_table(fp, columns=["episode_id", "dataset", "camera"]).to_pylist():
                    c = r["camera"]
                    hm = c.get("height_m")
                    f.write(json.dumps(dict(repo=repo, dataset=r["dataset"], episode_id=r["episode_id"], name=c["name"], width=c["width"],
                                            height=c["height"], K=[round(v, 4) for v in c["K"]], distortion_model=c["distortion_model"],
                                            distortion=[round(v, 6) for v in (c["distortion"] or [])],
                                            height_m=None if hm is None or not np.isfinite(hm) else round(float(hm), 3),
                                            height_source=c.get("height_source", "unknown"), camera_prompt=camera_prompt(c))) + "\n")
    with open(out / "eval_camera.jsonl", "w") as f:
        for version, root in (("v3", "/dev/shm/final_frame_eval"), ("v2", "/dev/shm/final_frame_eval_v2")):
            for fp in sorted(glob.glob(f"{root}/{version}_final_frame/test-*.parquet")):
                for r in pq.read_table(fp, columns=["suite_id", "scenario_id", "source", "camera"]).to_pylist():
                    cam = camera_calibration(r)
                    f.write(json.dumps(dict(suite=f"{version}_final_frame", suite_id=r["suite_id"], scenario_id=r["scenario_id"], source=r["source"],
                                            name=cam["name"], width=cam["width"], height=cam["height"], K=[round(x, 4) for x in cam["K"]],
                                            height_m=None if cam["height_m"] is None else round(cam["height_m"], 3),
                                            height_source=cam["height_source"], camera_prompt=camera_prompt(cam))) + "\n")
    print(f"calibration exported to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/dev/shm/camera_prov")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    work = Path(args.work)
    cameras = {}  # (repo, episode_id) -> camera dict with provenance
    for repo in EPISODE_REPOS:
        files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("data/episodes/")]
        folder = work / repo.split("/")[1]
        src = {}
        for f in files:
            local = hf_hub_download(repo, f, repo_type="dataset", local_dir=folder)
            rows = pq.read_table(local).to_pylist()
            if all("height_source" in r["camera"] for r in rows):  # already done (a rerun after a rate limit)
                for r in rows:
                    cameras[(repo, r["episode_id"])] = r["camera"]
                continue
            for r in rows:
                r["camera"] = with_source(r["camera"], r["dataset"])
                cameras[(repo, r["episode_id"])] = r["camera"]
            Dataset.from_list(rows, features=EPISODE_FEATURES).to_parquet(local)
            for r in rows:
                src[r["camera"]["height_source"]] = src.get(r["camera"]["height_source"], 0) + 1
        print(f"{repo.split('/')[1]}: {len(files)} episode files, height_source {src}", flush=True)
        if args.upload and src:  # one commit per repo (per-file commits hit the Hub's rate limit)
            upload(api, dict(repo_id=repo, repo_type="dataset", folder_path=str(folder / "data" / "episodes"), path_in_repo="data/episodes",
                             commit_message="episodes: camera.height_source (dataset | simulator | unknown)"))
    # dedup samples: the camera sentence per sample
    repo = EPISODE_REPOS[0]
    f = "data/samples/train-00000.parquet"
    local = hf_hub_download(repo, f, repo_type="dataset", local_dir=work / "dedup")
    t = pq.read_table(local)
    df = t.to_pandas()
    df["camera_prompt"] = [camera_prompt(cameras[(repo, e)]) for e in df["episode_id"]]
    import datasets
    feats = datasets.Features.from_arrow_schema(t.schema)
    feats["camera_prompt"] = datasets.Value("string")
    Dataset.from_pandas(df, features=feats, preserve_index=False).to_parquet(local)
    print("samples camera_prompt:", df["camera_prompt"].value_counts().to_dict())
    if args.upload:
        upload(api, dict(repo_id=repo, repo_type="dataset", folder_path=str(Path(local).parent), path_in_repo="data/samples",
                         commit_message="samples: camera_prompt (field of view and height from the stored calibration)"))
    # eval final-frame configs
    for version, root in (("v3", "/dev/shm/final_frame_eval"), ("v2", "/dev/shm/final_frame_eval_v2")):
        src_dir = Path(root) / f"{version}_final_frame"
        tmp = Path(root) / f"{version}_final_frame.new"
        tmp.mkdir(exist_ok=True)
        files = sorted(glob.glob(str(src_dir / "test-*.parquet")))
        counts = {}
        for fpath in files:
            rows = pq.read_table(fpath).to_pylist()
            for r in rows:
                cam = camera_calibration(r)
                r.update(camera_height_m=cam["height_m"], camera_height_source=cam["height_source"], camera_prompt=camera_prompt(cam))
                counts[(r["source"], r["camera_height_source"], r["camera_prompt"][-40:])] = counts.get((r["source"], r["camera_height_source"], r["camera_prompt"][-40:]), 0) + 1
            Dataset.from_list(rows, features=features()).to_parquet(tmp / Path(fpath).name)
        shutil.copy(src_dir / "missing.json", tmp / "missing.json")
        while subprocess.run(["pgrep", "-f", "vla.eval_final_frame"], capture_output=True).returncode == 0:
            time.sleep(20)  # never swap the tables while an evaluation reads them
        old = Path(root) / f"{version}_final_frame.old"
        shutil.rmtree(old, ignore_errors=True)
        src_dir.rename(old)
        tmp.rename(src_dir)
        shutil.rmtree(old, ignore_errors=True)
        print(f"{version}_final_frame: {len(files)} files; by source:", counts, flush=True)
        if args.upload:
            upload(api, dict(repo_id="Jinyan0924/qwen_robotics_nav_eval", repo_type="dataset", folder_path=str(src_dir),
                             path_in_repo=f"data/{version}_final_frame", commit_message=f"{version}_final_frame: camera calibration columns"))
    export_calibration(cameras, work)
    # the watcher's copy of v2 (built before future_images existed) gets the new table too
    live = Path("/dev/shm/final_frame_eval/v2_final_frame")
    if str(live) != "/dev/shm/final_frame_eval_v2/v2_final_frame":
        while subprocess.run(["pgrep", "-f", "vla.eval_final_frame"], capture_output=True).returncode == 0:
            time.sleep(20)
        shutil.rmtree(live, ignore_errors=True)
        shutil.copytree("/dev/shm/final_frame_eval_v2/v2_final_frame", live)
        print("replaced the local v2_final_frame used by the watcher")


if __name__ == "__main__":
    main()
