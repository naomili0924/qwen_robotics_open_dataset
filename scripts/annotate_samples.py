#!/usr/bin/env python
"""Describe what each deduplicated sample does, with a vision-language model (Claude).

For every kept sample the model sees 11 frames at 1 Hz, from 5 s before the current frame to 5 s after it, plus
the recorded path of the next 5 s as text (metres in the current frame, x forward, y left), and writes one or two
sentences: where the walker goes, what it passes, avoids or waits for, whether it stops or turns, and the place.
The descriptions are an auxiliary *output* for training (they are hindsight), never an input.

Frames come from the original source repos (the published dedup repo stores no images between +1 s and +4 s).

    # 1. pilot: a stratified handful, real-time requests, a parquet + an HTML sheet to look at
    python scripts/annotate_samples.py pilot --n 200
    # 2. the full set through the Batch API (half price): submit, then collect once the batches have ended
    python scripts/annotate_samples.py submit
    python scripts/annotate_samples.py collect
    # 3. upload data/annotations/train-00000.parquet to the dedup dataset repo
    python scripts/annotate_samples.py publish

Needs ANTHROPIC_API_KEY (and HF_TOKEN) in the environment, e.g. from /workspace/.env.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod.windows import FrameWindows, WindowConfig  # noqa: E402
from vla.stream import _Source, read_shard  # noqa: E402

MODEL = "claude-haiku-4-5"
HORIZON_S, N_WAY, PAST_S, FUTURE_S = 5.0, 10, 5, 5
MAX_SIDE, QUALITY = 512, 85
BATCH_BYTES = 200e6  # the API accepts 256 MB per batch
CARRIER = {"person_walking": "a person walking", "wheeled_robot": "a wheeled robot", "legged_robot": "a legged robot",
           "simulated_agent": "a simulated robot in a rendered house"}

SYSTEM = """You annotate egocentric navigation recordings for a robot-learning dataset.
You are shown 11 frames from a forward-facing camera, taken once per second from 5 s before "now" to 5 s after
"now", in order. You are also given the path the camera actually travelled during the 5 s after "now", as positions
in metres in the frame of the "now" view (x forward, y left), one every 0.5 s.

Describe, in one or two sentences, what the carrier of the camera does in the next 5 s and why, using only what is
visible in the frames and what follows from the path: direction (straight, bearing left/right, turning, turning
around, stopping, waiting), changes of speed, what is passed, avoided, followed or waited for (people, doors,
furniture, vehicles, stairs, crossings), and the kind of place. Do not invent things that are not visible. Do not quote
numbers; use words (stands still, slow, walking pace, bears left, turns sharply right). Prefer what explains the path:
the turn, the stop, the person stepped around, the door passed. Write for a reader who sees only the first six frames.

Reply with JSON only: {"description": "<1-2 sentences>", "place": "<2-4 words>", "interaction": "<what the carrier
reacts to, or 'none'>"}"""


def jpeg_b64(img):
    img = img.convert("RGB")
    s = MAX_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=QUALITY)
    return base64.standard_b64encode(buf.getvalue()).decode()


def path_text(target):
    xy = target[:, :2]
    seg = np.hypot(*np.diff(np.vstack([[0, 0], xy]), axis=0).T)
    pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in xy)
    bearing = np.degrees(np.arctan2(xy[-1, 1], xy[-1, 0])) if seg.sum() > 0.3 else 0.0
    side = "left" if bearing > 0 else "right"
    turn = ("goes straight ahead" if abs(bearing) < 15 else f"bears {side} by about {abs(bearing):.0f} degrees" if abs(bearing) < 45
            else f"turns sharply {side}, about {abs(bearing):.0f} degrees" if abs(bearing) < 120 else f"turns around to the {side}")
    v0, v1, vm = seg[:2].sum(), seg[-2:].sum(), seg.sum() / HORIZON_S
    pace = ("stands still" if vm < 0.15 else "moves very slowly" if vm < 0.5 else "moves at a slow walking pace" if vm < 1.0
            else "moves at walking pace" if vm < 1.8 else "moves fast")
    change = ("stops" if v1 < 0.15 < v0 else "starts moving" if v0 < 0.15 < v1 else "slows down" if v1 < v0 - 0.3
              else "speeds up" if v1 > v0 + 0.3 else "keeps a steady speed")
    return (f"Path for the next 5 s, one position every 0.5 s (x forward, y left, metres): {pts}. "
            f"In words: the carrier {pace}, {change}, and {turn} ({seg.sum():.1f} m in 5 s).")


def build_requests(repo, shard, rows_df):
    """One request per sample of this shard: 11 frames + the path text.  Yields (meta dict, params dict)."""
    src = _Source(repo, "train")
    episodes = src.load_episodes()
    ds = read_shard(repo, shard)
    w = FrameWindows(ds, episodes, WindowConfig(mode="final_frame", horizon_s=HORIZON_S, n_waypoints=N_WAY,
                                                n_past=PAST_S, past_dt_s=1.0, stride=1))
    col = ds.select_columns(["image"])
    for r in rows_df.itertuples():
        k = int(np.searchsorted(w.samples, r.row))
        if k >= len(w) or w.samples[k] != r.row:
            continue
        s = w.sample(k)
        i, e = int(r.row), w.ep_of_row[r.row]
        a, b = w.ep_start[e], w.ep_end[e]
        t0 = w.t[i]
        want = t0 + np.arange(-PAST_S, FUTURE_S + 1)
        rows = a + np.abs(w.t[a:b, None] - want[None]).argmin(0)
        with ThreadPoolExecutor(11) as pool:
            frames = list(pool.map(lambda j: jpeg_b64(col[int(j)]["image"]), rows))
        content = []
        for dt, f in zip(range(-PAST_S, FUTURE_S + 1), frames):
            label = "now" if dt == 0 else (f"{-dt} s before now" if dt < 0 else f"{dt} s after now")
            content.append({"type": "text", "text": f"Frame: {label}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f}})
        content.append({"type": "text", "text": f"The camera is carried by {CARRIER.get(s['embodiment'], 'a robot')}. "
                                                + path_text(s["target"])})
        custom_id = hashlib.sha1(f"{r.source}|{r.episode_id}|{int(r.frame_index)}".encode()).hexdigest()[:32]  # batch ids: [A-Za-z0-9_-]
        meta = dict(custom_id=custom_id, source=r.source, episode_id=r.episode_id, frame_index=int(r.frame_index),
                    bucket=r.bucket, embodiment=s["embodiment"], target=s["target"][:, :2].round(2).tolist(),
                    thumbs=frames if rows_df.attrs.get("keep_thumbs") else None)
        params = dict(model=MODEL, max_tokens=300, system=SYSTEM, messages=[{"role": "user", "content": content}])
        yield meta, params


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        return dict(description=str(d.get("description", "")).strip(), place=str(d.get("place", "")).strip(),
                    interaction=str(d.get("interaction", "")).strip(), raw=text)
    except Exception:
        return dict(description=text.strip(), place="", interaction="", raw=text)


def load_selection(args):
    sel = pd.read_parquet(args.selected)
    sel["source"] = sel["repo"].str.split("/").str[1]
    return sel


def pilot(args):
    import anthropic
    client = anthropic.Anthropic()
    sel = load_selection(args)
    rng = np.random.default_rng(0)
    # a few shards per source, then an even spread over the motion classes within them
    picked = []
    for repo, g in sel.groupby("repo"):
        shards = rng.choice(g["shard"].unique(), size=min(args.shards_per_source, g["shard"].nunique()), replace=False)
        gg = g[g["shard"].isin(shards)]
        per_source = max(1, round(args.n * np.sqrt(len(g)) / sum(np.sqrt(sel.groupby("repo").size()))))
        parts = [b.sample(min(len(b), max(1, per_source // gg["bucket"].nunique())), random_state=0) for _, b in gg.groupby("bucket")]
        picked.append(pd.concat(parts))
    picked = pd.concat(picked)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, usage = [], dict(input=0, output=0)

    def ask(meta, params):
        for attempt in range(5):
            try:
                r = client.messages.create(**params)
                break
            except anthropic.RateLimitError:
                time.sleep(10 * 2 ** attempt)
        else:
            raise RuntimeError("rate limited")
        text = next((b.text for b in r.content if b.type == "text"), "")
        usage["input"] += r.usage.input_tokens
        usage["output"] += r.usage.output_tokens
        return dict(meta, **parse(text), input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)

    for (repo, shard), g in picked.groupby(["repo", "shard"]):
        g.attrs["keep_thumbs"] = True
        reqs = list(build_requests(repo, shard, g))
        with ThreadPoolExecutor(args.concurrency) as pool:
            rows += list(pool.map(lambda mp: ask(*mp), reqs))
        print(f"{repo.split('/')[1]} {Path(shard).stem}: {len(reqs)} annotated, tokens so far {usage}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["thumbs"]).to_parquet(out / "pilot.parquet")
    n = len(df)
    print(f"{n} samples: {usage['input'] / n:.0f} input + {usage['output'] / n:.0f} output tokens each; "
          f"full set of {len(sel)} with the Batch API at Haiku 4.5 prices: "
          f"${len(sel) * (usage['input'] / n * 1.0 + usage['output'] / n * 5.0) / 1e6 / 2:.0f}")


def small(b64, side=224):
    from PIL import Image
    img = Image.open(io.BytesIO(base64.standard_b64decode(b64)))
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return base64.standard_b64encode(buf.getvalue()).decode()


def sheet(df, path):
    """An HTML page: for each sample the 11 frames, the path, and the description."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;margin:16px}"
             ".s{border-top:1px solid #ccc;padding:10px 0}.f{display:flex;gap:2px;overflow-x:auto}.f img{height:110px}"
             ".f .now img{outline:3px solid #d33}.c{font-size:11px;color:#666}.d{max-width:900px}</style></head><body>",
             f"<h2>Annotation pilot: {len(df)} samples, {MODEL}</h2>"]
    for r in df.itertuples():
        fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=80)
        xy = np.asarray(r.target)
        ax.plot(-xy[:, 1], xy[:, 0], "-o", ms=2, color="#d33")
        ax.plot(0, 0, "ks", ms=3)
        ax.set_xlim(-5, 5); ax.set_ylim(-1, 6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
        plot = base64.standard_b64encode(buf.getvalue()).decode()
        imgs = "".join(f"<span class='{'now' if dt == 0 else ''}'><img src='data:image/jpeg;base64,{small(f)}' title='{dt:+d} s'></span>"
                       for dt, f in zip(range(-PAST_S, FUTURE_S + 1), r.thumbs))
        parts.append(f"<div class='s'><div class='c'>{r.source} · {r.episode_id} · frame {r.frame_index} · {r.bucket} · {r.embodiment}</div>"
                     f"<div class='f'>{imgs}<img src='data:image/png;base64,{plot}'></div>"
                     f"<div class='d'><b>{r.description}</b><br><span class='c'>place: {r.place} · interaction: {r.interaction}</span></div></div>")
    Path(path).write_text("\n".join(parts) + "</body></html>")


def sheet_mode(args):
    """HTML sheet of --sheet annotated samples (frames read back from the sources), from any annotation table."""
    ann = pd.read_parquet(args.annotations)
    ann = ann[ann["description"].str.len() > 0] if "status" not in ann else ann[ann["status"] == "ok"]
    sel = load_selection(args).merge(ann[["source", "episode_id", "frame_index", "description", "place", "interaction"]],
                                     on=["source", "episode_id", "frame_index"])
    rng = np.random.default_rng(0)
    per = max(1, args.sheet // sel["source"].nunique())
    picked = []
    for _, g in sel.groupby("source"):  # the shard with most annotated samples, so few shards are downloaded
        top = g["shard"].value_counts().index[:2]
        gg = g[g["shard"].isin(top)]
        picked.append(gg.iloc[rng.permutation(len(gg))[:per]])
    picked = pd.concat(picked)
    rows = []
    for (repo, shard), g in picked.groupby(["repo", "shard"]):
        g.attrs["keep_thumbs"] = True
        by_id = g.set_index(["episode_id", "frame_index"])
        for meta, _ in build_requests(repo, shard, g):
            r = by_id.loc[(meta["episode_id"], meta["frame_index"])]
            rows.append(dict(meta, description=r["description"], place=r["place"], interaction=r["interaction"]))
    sheet(pd.DataFrame(rows), args.html)
    print(f"{len(rows)} samples on {args.html}")


def submit(args):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    sel = load_selection(args)
    out = Path(args.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    state_p = out / "batches.json"
    state = json.load(open(state_p)) if state_p.exists() else {"batches": {}, "done_shards": []}
    pending, size, n_sent = [], 0, 0

    def flush():
        nonlocal pending, size
        if not pending:
            return
        for attempt in range(6):
            try:
                b = client.messages.batches.create(requests=[Request(custom_id=m["custom_id"], params=MessageCreateParamsNonStreaming(**p))
                                                             for m, p in pending])
                break
            except anthropic.RateLimitError:
                time.sleep(30 * 2 ** attempt)
        else:
            raise RuntimeError("could not create a batch")
        state["batches"][b.id] = [m["custom_id"] for m, _ in pending]
        pd.DataFrame([m for m, _ in pending]).drop(columns=["thumbs"]).to_parquet(out / "meta" / f"{b.id}.parquet")
        json.dump(state, open(state_p, "w"), indent=1)
        print(f"batch {b.id}: {len(pending)} requests, {size / 1e6:.0f} MB", flush=True)
        pending, size = [], 0

    for (repo, shard), g in sel.groupby(["repo", "shard"]):
        key = f"{repo}|{shard}"
        if key in state["done_shards"]:
            continue
        for meta, params in build_requests(repo, shard, g):
            pending.append((meta, params))
            size += sum(len(c["source"]["data"]) for c in params["messages"][0]["content"] if c["type"] == "image")
            n_sent += 1
            if size >= BATCH_BYTES:
                flush()
        state["done_shards"].append(key)  # whole shards: the pending requests of this shard go with the next flush
        if args.max_samples and n_sent >= args.max_samples:
            break
    flush()
    print(f"submitted {n_sent} requests in {len(state['batches'])} batches")


def collect(args):
    import anthropic
    client = anthropic.Anthropic()
    out = Path(args.out)
    state = json.load(open(out / "batches.json"))
    (out / "results").mkdir(exist_ok=True)
    while True:
        open_ids = [b for b in state["batches"] if not (out / "results" / f"{b}.parquet").exists()]
        for b_id in open_ids:
            b = client.messages.batches.retrieve(b_id)
            if b.processing_status != "ended":
                continue
            meta = pd.read_parquet(out / "meta" / f"{b_id}.parquet").set_index("custom_id")
            rows = []
            for r in client.messages.batches.results(b_id):
                m = meta.loc[r.custom_id].to_dict()
                if r.result.type == "succeeded":
                    msg = r.result.message
                    text = next((bl.text for bl in msg.content if bl.type == "text"), "")
                    rows.append(dict(custom_id=r.custom_id, **m, **parse(text), status="ok",
                                     input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens))
                else:
                    rows.append(dict(custom_id=r.custom_id, **m, description="", place="", interaction="", raw="",
                                     status=r.result.type, input_tokens=0, output_tokens=0))
            pd.DataFrame(rows).to_parquet(out / "results" / f"{b_id}.parquet")
            print(f"{b_id}: {b.request_counts.succeeded} ok, {b.request_counts.errored} errored", flush=True)
        left = [b for b in state["batches"] if not (out / "results" / f"{b}.parquet").exists()]
        if not left or not args.wait:
            break
        print(f"{len(left)} batches still processing", flush=True)
        time.sleep(120)
    done = sorted((out / "results").glob("*.parquet"))
    if done:
        df = pd.concat([pd.read_parquet(p) for p in done], ignore_index=True)
        df["model"] = MODEL
        (out / "annotations").mkdir(exist_ok=True)
        df.to_parquet(out / "annotations" / "train-00000.parquet")
        ok = df["status"].eq("ok")
        print(f"{ok.sum()} annotated, {(~ok).sum()} failed; tokens {df.input_tokens.sum() / 1e6:.1f} M in, "
              f"{df.output_tokens.sum() / 1e6:.2f} M out")


def publish(args):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(Path(args.out) / "annotations"),
                      path_in_repo="data/annotations", commit_message="VLM descriptions of each sample (past 5 s + next 5 s)")
    print("uploaded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "submit", "collect", "publish", "sheet"])
    ap.add_argument("--selected", default="/dev/shm/dedup/selected/samples.parquet")
    ap.add_argument("--out", default="/dev/shm/dedup/annotate")
    ap.add_argument("--repo", default="Jinyan0924/qwen_robotics_nav_pretrain_dedup")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sheet", type=int, default=48, help="samples shown on the HTML sheet")
    ap.add_argument("--shards-per-source", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--annotations", default="/dev/shm/dedup/annotate/annotations/train-00000.parquet")
    ap.add_argument("--html", default="/dev/shm/dedup/annotate/sheet.html")
    args = ap.parse_args()
    globals()[args.mode if args.mode != "sheet" else "sheet_mode"](args)


if __name__ == "__main__":
    main()
