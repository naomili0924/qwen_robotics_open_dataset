"""Run a trained policy over a split, write predictions for scripts/evaluate.py and print the metrics.

    python -m vla.predict --checkpoint runs/x/last --split validation --out runs/x/val_predictions.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import eval as ev  # noqa: E402
from vla.data import NavDataset, columns_for, load_split, to_device  # noqa: E402
from vla.pipeline import NavigationPipeline  # noqa: E402
from vla.tasks import aux_tasks  # noqa: E402
from vla.train import make_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True, help="predictions JSON: {scenario_id: [[x, y] * horizon]}")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    pipe = NavigationPipeline.from_pretrained(args.checkpoint, batch_size=args.batch_size, workers=4)
    cfg, model = pipe.cfg, pipe.model
    model.eval()
    rows = load_split(cfg, args.split)
    tasks = aux_tasks(cfg.tasks)
    loader = make_loader(NavDataset(rows.select_columns(columns_for(cfg, tasks)), cfg, tasks), cfg, pipe.collate,
                         shuffle=False, limit=args.limit)
    preds, results, aux = {}, [], {t.name: ([], []) for t in tasks}
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, model.device)
            out = model.predict(batch)
            traj = out["trajectory"].cpu().numpy()
            for i, (sid, idx) in enumerate(zip(batch["scenario_id"], batch["index"].tolist())):
                preds[sid] = traj[i, :, :2].round(3).tolist()
                results.append(ev.evaluate_scenario(rows[idx], traj[i, :, :2]))
                for t in tasks:
                    aux[t.name][0].append(out[t.name][i].cpu().numpy())
                    aux[t.name][1].append(batch["aux"][t.name][i].cpu().numpy())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(args.out, "w"))
    metrics = ev.aggregate(results)
    for t in tasks:
        metrics.update(t.metrics(np.stack(aux[t.name][0]), np.stack(aux[t.name][1])))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
