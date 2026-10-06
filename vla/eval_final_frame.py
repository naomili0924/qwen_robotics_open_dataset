"""Score a final-frame policy on the evaluation suite, with the checks that it uses its inputs.

    python -m vla.eval_final_frame --hub-repo <user>/<repo> --checkpoint <run>/last --out ff_eval.json
    python -m vla.eval_final_frame --baselines-only --out baselines.json

Conditions scored: the true inputs; the final frame replaced by another scenario's ("swapped final frame");
the final frame replaced by the current frame ("no final frame"); the embodiment sentence swapped between
human and robot ("swapped prompt").  If the model uses the final frame, completion must drop in the second
and third; if it uses the prompt, predicted speed must change in the fourth.
"""
import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import suite  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402
from hnod.windows import embodiment_prompt  # noqa: E402


def load_rows(args):
    import pyarrow.parquet as pq
    if args.data:
        files = sorted(glob.glob(os.path.join(args.data, "test-*.parquet")))
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(args.repo, repo_type="dataset", allow_patterns=[f"data/{args.version}/*"],
                              local_dir="/dev/shm/final_frame_eval/hub")
        files = sorted(glob.glob(f"{d}/data/{args.version}/test-*.parquet"))
    return [r for f in files for r in pq.read_table(f).to_pylist()]


def _img(cell):
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def row_camera(r):
    """Camera dict of an eval row for the prompt (field of view from K; height only when its source is certain)."""
    cam = dict(r.get("camera") or {})
    cam.update(height_m=r.get("camera_height_m"), height_source=r.get("camera_height_source"))
    return cam


def build_items(rows, cfg, final="true", prompt="true", camera="true"):
    from vla.data import final_frame_item
    items = []
    for i, r in enumerate(rows):
        back = np.round(np.arange(cfg.frames - 2, -1, -1) * cfg.past_dt_s * r["rate_hz"]).astype(int)
        past = [_img(r["past_images"][j]) for j in np.clip(CURRENT - back, 0, CURRENT)]
        if final == "true":
            fin = _img(r["final_image"])
        elif final == "swapped":  # the final frame of a scenario from a different recording
            j = next(k for k in list(range(i + 1, len(rows))) + list(range(i)) if rows[k]["sequence"] != r["sequence"])
            fin = _img(rows[j]["final_image"])
        else:
            fin = past[-1]
        emb = r["embodiment"]
        if prompt == "swapped":
            emb = "wheeled_robot" if emb == "person_walking" else "person_walking"
        cam = None
        if cfg.camera_prompt:  # "unknown": the sentence says field of view and height are unknown (does the model use it?)
            cam = row_camera(r) if camera == "true" else {}
        items.append(final_frame_item(past, fin, embodiment_prompt(emb, cfg.past_dt_s, cfg.horizon, cfg.horizon_s, camera=cam),
                                      cfg, index=i, scenario_id=r["suite_id"]))
    return items


@torch.no_grad()
def predict(pipe, items, batch_size):
    from vla.data import to_device
    out = []
    for i in range(0, len(items), batch_size):
        batch = to_device(pipe.collate(items[i:i + batch_size]), pipe.model.device)
        out.append(pipe.model.predict(batch)["trajectory"][:, :, :2].float().cpu().numpy())
    return np.concatenate(out)


def table(rows, results):
    out = {"all": suite.aggregate(results, suite.TIMED_METRICS)}
    for key in ("environment", "source", "embodiment"):
        for v in sorted({r[key] for r in rows}):
            sub = [x for x, r in zip(results, rows) if r[key] == v]
            out[f"{key}:{v}"] = suite.aggregate(sub, suite.TIMED_METRICS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--hub-repo", default="")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_eval")
    ap.add_argument("--version", default="v2_final_frame")
    ap.add_argument("--data", default="", help="local directory with the suite parquet files")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_rows(args)
    report = {"scenarios": len(rows), "version": args.version, "conditions": {}}
    for name in ("recorded", "constant_velocity", "stationary"):
        report["conditions"][f"baseline: {name}"] = table(rows, [suite.score_timed(r, suite.timed_baselines(r)[name]) for r in rows])
    preds = {}
    if not args.baselines_only:
        from vla.pipeline import NavigationPipeline
        opts = dict(batch_size=args.batch_size, workers=0, stream=False)
        pipe = (NavigationPipeline.from_hub(args.hub_repo, args.checkpoint, args.revision, **opts) if args.hub_repo
                else NavigationPipeline.from_pretrained(args.checkpoint, **opts))
        pipe.model.eval()
        conditions = [("model", {}), ("model, swapped final frame", dict(final="swapped")),
                      ("model, no final frame", dict(final="none")), ("model, swapped prompt", dict(prompt="swapped"))]
        if pipe.cfg.camera_prompt:  # does the model use the calibration sentence?
            conditions.append(("model, unknown camera", dict(camera="unknown")))
        for name, kw in conditions:
            P = predict(pipe, build_items(rows, pipe.cfg, **kw), args.batch_size)
            preds[name] = P
            report["conditions"][name] = table(rows, [suite.score_timed(r, p) for r, p in zip(rows, P)])
        base = preds["model"]
        report["input_use"] = {k: float(np.linalg.norm(preds[k][:, -1] - base[:, -1], axis=1).mean())
                               for k in preds if k != "model"}  # mean shift of the predicted end point, metres
        report["checkpoint"] = args.checkpoint
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(report, predictions={k: {r["suite_id"]: p.round(3).tolist() for r, p in zip(rows, v)}
                                        for k, v in preds.items()}), open(args.out, "w"), default=float)
    cols = ["success", "completed", "collided", "ade", "end_error_m", "speed_mps", "wiggle_rad"]
    print(f"{len(rows)} scenarios ({args.version})")
    print(f"{'':34s}" + "".join(f"{c[:11]:>12s}" for c in cols))
    for name, t in report["conditions"].items():
        print(f"{name:34s}" + "".join(f"{t['all'][c]:12.3f}" for c in cols))
    for grp in ("environment:indoor", "environment:outdoor", "embodiment:person_walking", "embodiment:wheeled_robot"):
        if "model" in report["conditions"] and grp in report["conditions"]["model"]:
            t = report["conditions"]["model"][grp]
            print(f"{'  model, ' + grp.split(':')[1]:34s}" + "".join(f"{t[c]:12.3f}" for c in cols) + f"   n={t['num_scenarios']}")
    if "input_use" in report:
        print("shift of the predicted end point when an input is changed (m):", {k: round(v, 2) for k, v in report["input_use"].items()})


if __name__ == "__main__":
    main()
