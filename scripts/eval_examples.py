#!/usr/bin/env python
"""Example sheet of a final-frame evaluation: per scenario the views (1 Hz past, now, final), the bird's-eye view with
the recorded path, the model's prediction and the constant-velocity baseline, the metrics and the description.

    python scripts/eval_examples.py --eval /dev/shm/runs/<run>/eval/<file>.json --version v3 --out sheet.html
"""
import argparse
import base64
import glob
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnod import suite  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402
from visualize import draw  # noqa: E402


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    return base64.standard_b64encode(buf.getvalue()).decode()


def thumb(cell, side=200):
    im = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    return base64.standard_b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval JSON of vla.eval_final_frame (with predictions)")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--eval-dir", default="/dev/shm/final_frame_eval")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", default="promising", choices=["promising", "failures", "random"])
    ap.add_argument("--out", default="/dev/shm/eval_examples.html")
    ap.add_argument("--title", default="Final-frame model on v3")
    ap.add_argument("--text-is-input", action="store_true", help="the description was the model's task text (instruction policies)")
    args = ap.parse_args()
    ev = json.load(open(args.eval))
    pred = ev["predictions"]["model"]
    rows = [r for f in sorted(glob.glob(f"{args.eval_dir}/{args.version}_final_frame/test-*.parquet")) for r in pq.read_table(f).to_pylist()]
    ann = {}
    for f in glob.glob(f"{args.eval_dir}/{args.version}_annotations/test-*.parquet"):
        ann.update({r["suite_id"]: r for r in pq.read_table(f).to_pylist()})
    scored = []
    for r in rows:
        if r["suite_id"] not in pred:
            continue
        p = np.asarray(pred[r["suite_id"]])
        m = suite.score_timed(r, p)
        cv = suite.score_timed(r, suite.timed_baselines(r)["constant_velocity"])
        motion = next((t.split(":")[1] for t in r["tags"] if t.startswith("motion:")), "")
        scored.append(dict(row=r, pred=p, m=m, cv=cv, motion=motion))
    rng = np.random.default_rng(0)
    if args.mode == "promising":  # the model completes, the baseline does not; prefer non-straight motion and lower FDE
        pool = [s for s in scored if s["m"]["success"] and not s["cv"]["success"]]
        pool.sort(key=lambda s: (s["motion"] == "straight", s["m"]["fde"]))
    elif args.mode == "failures":
        pool = sorted([s for s in scored if not s["m"]["success"]], key=lambda s: -s["m"]["fde"])
    else:
        pool = list(rng.permutation(scored))
    picked, seen = [], {}
    for s in pool:  # spread over motion classes
        if seen.get(s["motion"], 0) >= max(2, args.n // 4):
            continue
        picked.append(s); seen[s["motion"]] = seen.get(s["motion"], 0) + 1
        if len(picked) >= args.n:
            break
    parts = ["<html><head><meta charset='utf-8'><title>" + args.title + "</title><style>body{font-family:sans-serif;margin:16px}"
             ".s{border-top:1px solid #ccc;padding:12px 0}.f{display:flex;gap:3px;flex-wrap:wrap;align-items:flex-start}"
             ".f img{height:150px}.f .fin img{outline:3px solid #d33}.c{font-size:12px;color:#555}.d{max-width:1000px;margin-top:6px}"
             "table{border-collapse:collapse;font-size:12px}td,th{padding:2px 8px;border-bottom:1px solid #eee;text-align:right}</style></head><body>",
             f"<h2>{Path(args.eval).stem}: {args.mode} examples on {args.version}_final_frame "
             f"({len([s for s in scored if s['m']['success']])} of {len(scored)} completed; constant velocity {len([s for s in scored if s['cv']['success']])})</h2>",
             "<p class='c'>" + ("The model saw the past views and now, plus the task text below; the final view (red frame) is shown for reference only. " if args.text_is_input else "") + "Views: 1 Hz past, <b>now</b>, then the final view (red frame). Bird's-eye view: static map grey, people red with their "
             "future tracks, recorded path green, <b>model red</b>, constant velocity dashed; x forward (up), y left.</p>"]
    for s in picked:
        r, p = s["row"], s["pred"]
        ts = np.asarray(r["timestamps"]); t0 = ts[CURRENT]
        idx = [int(np.argmin(np.abs(ts[:CURRENT + 1] - (t0 - sec)))) for sec in range(5, 0, -1)] + [CURRENT]
        imgs = "".join(f"<span><img src='data:image/jpeg;base64,{thumb(r['past_images'][i])}' title='{ts[i]-t0:+.1f} s'></span>" for i in idx)
        imgs += f"<span class='fin'><img src='data:image/jpeg;base64,{thumb(r['final_image'])}' title='final, +{r['horizon_s']:g} s'></span>"
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        draw(ax, r, view=float(np.clip(np.abs(p).max() + 2.0, 5, 10)))
        cvp = suite.timed_baselines(r)["constant_velocity"]
        ax.plot(np.r_[0, cvp[:, 1]], np.r_[0, cvp[:, 0]], "--", color="#555", lw=1.2, label="constant velocity")
        ax.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], ".-", color="#d33", lw=2, ms=5, label="model")
        ax.plot(r["final_xy"][1], r["final_xy"][0], "*", color="#2ca02c", ms=12, label="recorded end")
        ax.legend(fontsize=7, loc="lower right"); ax.set_title(f"{r['suite_id']}  {s['motion']}", fontsize=9)
        m, cv = s["m"], s["cv"]
        a = ann.get(r["suite_id"], {})
        parts.append(
            f"<div class='s'><div class='c'>{r['suite_id']} · {r['source']} · {r['environment']} · {r['embodiment']} · motion: <b>{s['motion']}</b> · horizon {r['horizon_s']:g} s</div>"
            f"<div class='f'>{imgs}<img src='data:image/png;base64,{png(fig)}' style='height:300px'></div>"
            f"<table><tr><th></th><th>completed</th><th>collided</th><th>ADE</th><th>FDE</th></tr>"
            f"<tr><td>model</td><td>{'yes' if m['success'] else 'no'}</td><td>{'yes' if m['collided'] else 'no'}</td><td>{m['ade']:.2f}</td><td>{m['fde']:.2f}</td></tr>"
            f"<tr><td>constant velocity</td><td>{'yes' if cv['success'] else 'no'}</td><td>{'yes' if cv['collided'] else 'no'}</td><td>{cv['ade']:.2f}</td><td>{cv['fde']:.2f}</td></tr></table>"
            f"<div class='d'><b>{'Task text given to the model' if args.text_is_input else 'Description'}:</b> {a.get('description', '')}<br><span class='c'>place: {a.get('place', '')} · interaction: {a.get('interaction', '')}</span></div></div>")
    Path(args.out).write_text("\n".join(parts) + "</body></html>")
    print(f"{len(picked)} examples -> {args.out}; motion classes: {seen}")


if __name__ == "__main__":
    main()
