"""Run a trained policy on the evaluation suite and score it (hnod.suite).

    python -m vla.predict_suite --checkpoint runs/x/last --out runs/x/suite_v1.json

Works for policies trained on scenario rows (data_format=scenarios) and on per-frame data
(data_format=frames): the suite row is turned into the same inputs the policy saw in training.
The predicted waypoints are treated as a path that the robot follows at the suite speed.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import suite  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402
from vla.data import NavDataset, frames_prompt, to_device  # noqa: E402
from vla.pipeline import NavigationPipeline  # noqa: E402
from vla.train import make_loader  # noqa: E402


class SuiteDataset(Dataset):
    def __init__(self, rows, cfg):
        self.rows, self.cfg = rows, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, cfg = self.rows[i], self.cfg
        if cfg.data_format != "frames":
            item = NavDataset(self.rows, cfg)[i]
            item["scenario_id"] = row["suite_id"]
            return item
        images = [im.convert("RGB") for im in row["past_images"][-cfg.frames:]]
        e = row["ego"]
        past = np.stack([e["x"], e["y"]], 1)[:CURRENT + 1] - [e["x"][CURRENT], e["y"][CURRENT]]
        s = dict(past_xy=past.astype(np.float32), velocity=np.array([e["vx"][CURRENT], e["vy"][CURRENT]], np.float32),
                 prompt=row["prompt"])
        kin = np.r_[past.ravel(), s["velocity"], row["goal"]].astype(np.float32)
        return dict(index=i, scenario_id=row["suite_id"], images=images, prompt=frames_prompt(s, cfg), kin=kin,
                    target=np.zeros((cfg.horizon, cfg.action_dim), np.float32), aux={})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_eval")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--data", help="local directory of suite parquet files instead of the Hub")
    ap.add_argument("--out", required=True, help="predictions JSON {suite_id: [[x, y], ...]}")
    ap.add_argument("--speed", type=float, default=suite.ROBOT_SPEED)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset
    if args.data:
        rows = load_dataset("parquet", data_files=sorted(glob.glob(os.path.join(args.data, "test-*.parquet"))),
                            split="train")
    else:
        rows = load_dataset(args.repo, args.version, split="test")
    pipe = NavigationPipeline.from_pretrained(args.checkpoint, batch_size=args.batch_size, workers=4)
    cfg, model = pipe.cfg, pipe.model
    model.eval()
    loader = make_loader(SuiteDataset(rows, cfg), cfg, pipe.collate, shuffle=False)
    preds, results = {}, []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, model.device)
            traj = model.predict(batch)["trajectory"].cpu().numpy()
            for i, (sid, idx) in enumerate(zip(batch["scenario_id"], batch["index"].tolist())):
                path = traj[i, :, :2]
                preds[sid] = path.round(3).tolist()
                results.append(suite.score(rows[idx], path, speed=args.speed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(args.out, "w"))
    print(json.dumps(suite.aggregate(results), indent=2))


if __name__ == "__main__":
    main()
