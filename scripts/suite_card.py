#!/usr/bin/env python
"""Dataset card, distribution figure and baseline table for a written evaluation suite.

    python scripts/suite_card.py --suite /workspace/cache/suite/suite/v1 --out-card dataset_card_eval.md
"""
import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import suite  # noqa: E402

GROUPS = [("env", "Environment"), ("source", "Source"), ("scene", "Scene type (CLIP, audited)"),
          ("people", "People (recorded tracks)"), ("path", "Reference path"), ("goal", "Goal bearing"),
          ("layout", "Layout"), ("lighting", "Lighting"), ("baseline_fails", "Naive baselines that fail")]
BASELINES = ["reference", "stationary", "straight_ahead", "straight_to_goal"]


def distribution(rows):
    tags = Counter(t for r in rows for t in r["tags"])
    out = {}
    for g, _ in GROUPS:
        out[g] = {t.split(":", 1)[1]: n for t, n in sorted(tags.items(), key=lambda kv: -kv[1]) if t.split(":")[0] == g}
    return out


def plot_distribution(dist, n, path):
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (g, title) in zip(axs.ravel(), GROUPS):
        d = dist.get(g, {})
        ax.barh(list(d)[::-1], list(d.values())[::-1], color="#4c72b0")
        for y, v in enumerate(list(d.values())[::-1]):
            ax.text(v, y, f" {v}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(d.values(), default=1) * 1.2)
    fig.suptitle(f"Scenario distribution ({n} scenarios; a scenario can carry several people / baseline tags)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def baseline_table(rows):
    out = {}
    for b in BASELINES:
        res = []
        for r in rows:
            if b == "reference":
                e = r["ego"]
                ref = np.stack([e["x"], e["y"]], 1)[suite.CURRENT + 1:]
                p = ref - np.array([e["x"][suite.CURRENT], e["y"][suite.CURRENT]])
            else:
                p = suite.baseline_paths(r)[b]
            x = suite.score(r, p)
            x["env"] = r["environment"]
            res.append(x)
        out[b] = {env: suite.aggregate([x for x in res if env in ("all", x["env"])]) for env in ("all", "indoor", "outdoor")}
    return out


def example_grid(audit_dir, ids, path):
    import matplotlib.image as mpimg
    fig, axs = plt.subplots(len(ids), 1, figsize=(15, 6.5 * len(ids)))
    for ax, sid in zip(np.atleast_1d(axs), ids):
        ax.imshow(mpimg.imread(os.path.join(audit_dir, f"{sid}.png")))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="written suite directory (data/, audit/)")
    ap.add_argument("--out-card", default="dataset_card_eval.md")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--examples", nargs="*", default=[])
    args = ap.parse_args()
    rows = list(load_dataset("parquet", data_files=sorted(glob.glob(f"{args.suite}/data/test-*.parquet")),
                             split="train"))
    dist = distribution(rows)
    Path(args.figures).mkdir(exist_ok=True)
    plot_distribution(dist, len(rows), f"{args.figures}/eval_distribution.png")
    if args.examples:
        example_grid(f"{args.suite}/audit", args.examples, f"{args.figures}/eval_examples.png")
    table = baseline_table(rows)
    import json
    json.dump(dict(distribution=dist, baselines=table), open(f"{args.suite}/summary.json", "w"), indent=1,
              default=float)
    print(json.dumps(dict(distribution=dist), indent=1))
    for b, t in table.items():
        print(b, {env: {k: round(v, 3) for k, v in m.items() if k in ("success", "collided", "progress_ratio",
                                                                         "reference_deviation_m")} for env, m in t.items()})


if __name__ == "__main__":
    main()
