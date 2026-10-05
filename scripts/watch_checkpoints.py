#!/usr/bin/env python
"""Publish milestone and best checkpoints of a running training to the Hub, next to its `last/`.

Runs beside vla.train (no restart needed).  For run <run> and model repo <hub_repo> it keeps on the Hub:

* <run name>/step_<N>/ for every N that is a multiple of --every (LoRA, heads, config; no optimiser state);
* <run name>/best/: the checkpoint with the lowest validation loss so far, with best.json (step, val_loss).
  Validation loss is the run's own held-out loss (log.jsonl), never the evaluation suite;
* with --suite: suite metrics of every milestone under <run name>/eval/ (reporting only).

    python scripts/watch_checkpoints.py --run /workspace/runs/e1_all_sqrt --hub-repo <user>/<repo> --every 5000
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

WEIGHTS = ("head.pt", "aux_heads.pt", "config.json", "lora")


def snapshot(step_dir, tmp):
    """Copy the weights of a finished step folder (trainer.pt marks it complete); None if it vanished."""
    try:
        if not (step_dir / "trainer.pt").exists():
            return None
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name in WEIGHTS:
            src = step_dir / name
            if src.is_dir():
                shutil.copytree(src, tmp / name)
            elif src.exists():
                shutil.copy(src, tmp / name)
        return tmp
    except FileNotFoundError:  # pruned while copying
        return None


def val_losses(run):
    out = {}
    for line in open(run / "log.jsonl"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_loss" in d:
            out[int(d["step"])] = float(d["val_loss"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hub-repo", required=True)
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--suite", nargs="*", default=[], help="suite versions to score at each milestone, e.g. v1 v2")
    ap.add_argument("--suite-data", default="/workspace/cache/suite/suite/{version}/data")
    ap.add_argument("--poll", type=float, default=60)
    args = ap.parse_args()
    run, api = Path(args.run), HfApi(token=os.environ.get("HF_TOKEN"))
    state_path = run / "watcher_state.json"
    state = json.load(open(state_path)) if state_path.exists() else dict(milestones=[], best=None, seen=[])
    tmp = Path("/dev/shm/ckpt_watch") / run.name

    def save_state():
        json.dump(state, open(state_path, "w"), indent=1)

    while True:
        try:
            losses = val_losses(run)
            for d in sorted(run.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
                step = int(d.name.split("_")[1])
                if step in state["seen"]:
                    continue
                snap = snapshot(d, tmp / d.name)
                if snap is None:
                    continue
                if step % args.every == 0:
                    api.upload_folder(repo_id=args.hub_repo, folder_path=str(snap), path_in_repo=f"{run.name}/{d.name}",
                                      commit_message=f"{run.name}: milestone {d.name}")
                    state["milestones"].append(step)
                    print(time.strftime("%H:%M:%S"), "milestone", step, flush=True)
                    for v in args.suite:
                        out = run / "eval" / f"{d.name}_{v}.json"
                        out.parent.mkdir(exist_ok=True)
                        r = subprocess.run([sys.executable, "-m", "vla.predict_suite", "--checkpoint", str(snap),
                                            "--version", v, "--data", args.suite_data.format(version=v),
                                            "--batch-size", "2", "--out", str(out)], capture_output=True, text=True)
                        if r.returncode == 0:
                            for f in (out, Path(str(out).replace(".json", "_metrics.json"))):
                                api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{run.name}/eval/{f.name}",
                                                repo_id=args.hub_repo, commit_message=f"{run.name}: suite {v} at {d.name}")
                            print(time.strftime("%H:%M:%S"), "suite", v, "scored at", step, flush=True)
                        else:
                            print("suite eval failed:", r.stderr[-300:], flush=True)
                if step in losses and (state["best"] is None or losses[step] < state["best"]["val_loss"]):
                    json.dump(dict(step=step, val_loss=losses[step]), open(snap / "best.json", "w"))
                    api.upload_folder(repo_id=args.hub_repo, folder_path=str(snap), path_in_repo=f"{run.name}/best",
                                      commit_message=f"{run.name}: best so far, step {step}, val_loss {losses[step]:.5f}",
                                      delete_patterns=["*"])
                    state["best"] = dict(step=step, val_loss=losses[step])
                    print(time.strftime("%H:%M:%S"), "best", state["best"], flush=True)
                state["seen"].append(step)
                save_state()
                shutil.rmtree(snap, ignore_errors=True)
        except Exception as e:  # network problems must not end the watcher
            print(time.strftime("%H:%M:%S"), "error:", repr(e)[:300], flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
