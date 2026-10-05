#!/usr/bin/env python
"""Build the evaluation suite (docs/eval_design.md) from held-out splits of the published scenario sets.

Pass 1 (candidates): every row of the held-out splits is checked and described: the recorded
reference path must be collision-free and mostly observed; a goal is placed on the reference path
beyond the scored horizon, using later rows of the same recording; the scenario is tagged
(indoor / outdoor and scene type by CLIP, people, path shape, goal bearing, narrow passages,
lighting) and the naive baselines are scored in the suite protocol.  Results go to
<work>/candidates.parquet.
Pass 2 (select): quotas per environment and source, tag coverage first, then scenarios where the
baselines fail; at most a few scenarios per recording, spaced in time.
Pass 3 (write): the selected rows, with `goal` replaced by the new goal and prompt / tag columns
added, to <work>/suite/<version>/test-*.parquet, plus one audit image per scenario.

    python scripts/build_eval_suite.py candidates
    python scripts/build_eval_suite.py select --indoor 100 --outdoor 50
    python scripts/build_eval_suite.py write
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnod import eval as ev  # noqa: E402
from hnod import suite  # noqa: E402
from hnod.scenario import CURRENT, N_FUTURE  # noqa: E402

# held-out splits only: none of these is used for training (training uses each source's train split)
SOURCES = [
    dict(name="coda", repo="Jinyan0924/qwen_robotics_open_dataset", config="coda_2hz", split="test",
         reference="teleoperated_robot", environment=None),
    dict(name="jrdb", repo="Jinyan0924/qwen_robotics_open_dataset_jrdb", config="jrdb_2.5hz", split="test",
         reference="teleoperated_robot", environment=None),
    dict(name="robosense", repo="Jinyan0924/qwen_robotics_open_dataset_robosense", config="robosense_1hz",
         split="validation", reference="teleoperated_robot", environment=None),
    dict(name="hssd", repo="Jinyan0924/habitat_hssd_pointgoal_nav_scenarios", config="hssd_2hz", split="test",
         reference="shortest_path_planner", environment="indoor"),
    # real indoor scenes are scarce, so the validation splits of CODa and JRDB are reserved for evaluation too
    dict(name="coda_val", repo="Jinyan0924/qwen_robotics_open_dataset", config="coda_2hz", split="validation",
         reference="teleoperated_robot", environment=None),
    dict(name="jrdb_val", repo="Jinyan0924/qwen_robotics_open_dataset_jrdb", config="jrdb_2.5hz",
         split="validation", reference="teleoperated_robot", environment=None),
    # JRDB (indoor crowds with lidar and tracks) is reserved for evaluation as a whole: it adds only about 1,300
    # frames to training but most of the real indoor scenarios with people
    dict(name="jrdb_train", repo="Jinyan0924/qwen_robotics_open_dataset_jrdb", config="jrdb_2.5hz", split="train",
         reference="teleoperated_robot", environment=None),
    # more simulated houses: HSSD's validation houses (training uses its train houses only)
    dict(name="hssd_val", repo="Jinyan0924/habitat_hssd_pointgoal_nav_scenarios", config="hssd_2hz",
         split="validation", reference="shortest_path_planner", environment="indoor"),
]
LIGHT = ["scenario_id", "sequence", "segment", "source_frames", "world_from_scenario", "ego"]
GOAL_EXTRA_M = (3.0, 10.0)   # goal: this much further along the reference than the end of the horizon
MIN_EXTRA_M = 1.0            # ... at least this much path must exist beyond the horizon
INDOOR_SCENES = {"corridor": "a photo of a long corridor or hallway inside a building",
                 "doorway": "a photo of a doorway or an open door",
                 "lobby": "a photo of a large open lobby or atrium",
                 "room": "a photo of a room with furniture",
                 "dining": "a photo of a cafeteria or dining area",
                 "kitchen": "a photo of a kitchen",
                 "stairs": "a photo of a staircase",
                 "office": "a photo of an office with desks",
                 "hall_crowd": "a photo of a crowded indoor hall with many people"}
OUTDOOR_SCENES = {"sidewalk": "a photo of a sidewalk", "crossing": "a photo of a street crossing",
                  "plaza": "a photo of a plaza or courtyard", "path": "a photo of a footpath in a park",
                  "road": "a photo of a road with cars", "entrance": "a photo of a building entrance outside"}


def _world_xy(row):
    T = np.asarray(row["world_from_scenario"], float).reshape(4, 4)
    e = row["ego"]
    p = np.c_[e["x"], e["y"], np.zeros(len(e["x"])), np.ones(len(e["x"]))] @ T.T
    return p[:, :2]


def _thumb(img_bytes, width=336):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h = int(round(img.shape[0] * width / img.shape[1]))
    small = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small)
    return buf.tobytes(), float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def light_rows(path):
    """Worker: poses only, for deriving goals."""
    t = pq.read_table(path, columns=LIGHT).to_pylist()
    return [dict(scenario_id=r["scenario_id"], sequence=str(r["sequence"]), segment=int(r["segment"]),
                 frame=int(r["source_frames"][CURRENT]), source_frames=list(r["source_frames"]),
                 world_xy=_world_xy(r).tolist(), T=list(r["world_from_scenario"])) for r in t]


def derive_goals(rows, extra_m=GOAL_EXTRA_M, min_extra=MIN_EXTRA_M, until_end=False, seed=0):
    """{scenario_id: (goal xy in the scenario frame, metres beyond the horizon)} for one source.

    The goal lies on the reference path `extra_m` beyond the end of the scored horizon, built from
    the world positions of every row of the same recording.  until_end: episodes end at their goal
    (simulator), so a goal closer than that is the episode end itself.
    """
    rng = np.random.default_rng(seed)
    track = defaultdict(dict)
    for r in rows:
        for f, xy in zip(r["source_frames"], r["world_xy"]):
            track[(r["sequence"], r["segment"])].setdefault(int(f), xy)
    out = {}
    for r in rows:
        tr = track[(r["sequence"], r["segment"])]
        fs = np.array(sorted(f for f in tr if f >= r["frame"]))
        W = np.array([tr[f] for f in fs])
        T = np.asarray(r["T"], float).reshape(4, 4)
        local = (np.c_[W, np.zeros(len(W)), np.ones(len(W))] @ np.linalg.inv(T).T)[:, :2]
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(local, axis=0).T))]
        s_h = s[min(np.searchsorted(fs, r["source_frames"][-1]), len(s) - 1)]
        want = s_h + rng.uniform(*extra_m)
        if s[-1] - s_h < min_extra and not until_end:
            continue
        target = min(want, s[-1])
        out[r["scenario_id"]] = ([float(np.interp(target, s, local[:, 0])), float(np.interp(target, s, local[:, 1]))],
                                 float(target - s_h))
    return out


def scan_shard(path, source, goals):
    """Worker: reference checks, baselines with the derived goal, people tags and a thumbnail per row."""
    out = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if row["scenario_id"] not in goals:
                continue
            goal, extra = goals[row["scenario_id"]]
            row["goal"] = goal
            ref = np.stack([row["ego"]["x"], row["ego"]["y"]], 1)[CURRENT:]
            r = ev.evaluate_scenario(row, ref[1:])
            clearance = min(r["clearance_map"], r["clearance_static"])
            ptags, n_people = suite.people_tags(row, ref)
            base = {k: suite.score(row, p) for k, p in suite.baseline_paths(row).items()}
            # the reference under the suite protocol (followed at the robot speed): a scenario must be solvable as scored
            rel = ref[1:] - ref[0]
            proto = suite.score(row, rel)
            heading0 = rel[min(1, len(rel) - 1)]
            reverses = bool(rel[0, 0] < -0.05 or heading0[0] < -0.1)  # the recording platform backs up
            img = row["past_images"][-1]["bytes"] if row.get("past_images") else None
            thumb, lum = _thumb(img) if img else (None, float("nan"))
            rec = dict(
                scenario_id=row["scenario_id"], source=source["name"], shard=str(path), sequence=str(row["sequence"]),
                segment=int(row["segment"]), rate_hz=float(row["rate_hz"]), frame=int(row["source_frames"][CURRENT]),
                time_s=float(row["timestamps"][CURRENT]), goal=goal, goal_extra_m=extra,
                ref_collided=bool(r["collided"]), ref_collided_lidar=bool(r["collided_lidar"]),
                ref_unknown=float(r["unknown_fraction"]), start_overlap=bool(r["start_overlap"]),
                ref_clearance=float(clearance), people_tags=ptags, n_people=n_people,
                ref_length=float(np.hypot(*np.diff(ref, axis=0).T).sum()),
                path_tags=suite.path_tags(ref, np.asarray(goal), clearance),
                luminance=lum, thumb=thumb, has_map=row.get("static_map") is not None,
                has_lidar=row.get("future_lidar") is not None,
                ref_protocol_collided=bool(proto["collided"]), ref_protocol_progress=float(proto["progress_ratio"]),
                ref_reverses=reverses)
            for k, b in base.items():
                rec[f"{k}_collided"] = bool(b["collided"])
                rec[f"{k}_success"] = bool(b["success"])
                rec[f"{k}_progress"] = float(b["progress_ratio"])
            out.append(rec)
    return out


def cmd_candidates(args):
    work = Path(args.work)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    recs = []
    for src in SOURCES:
        if args.sources and src["name"] not in args.sources:
            continue
        files = sorted(f for f in api.list_repo_files(src["repo"], repo_type="dataset")
                       if f.startswith(f"data/{src['config']}/{src['split']}-"))
        if args.max_shards:
            files = files[:args.max_shards]
        paths = [hf_hub_download(src["repo"], f, repo_type="dataset", local_dir=work / "src" / src["name"])
                 for f in files]
        with ProcessPoolExecutor(args.workers) as pool:
            light = [r for rows in pool.map(light_rows, paths) for r in rows]
            goals = derive_goals(light, until_end=dataset_of(src["name"]) == "hssd", seed=args.seed)
            n0 = len(recs)
            for out in pool.map(scan_shard, paths, [src] * len(paths), [goals] * len(paths)):
                recs += out
        print(f"{src['name']}: {len(paths)} shards, {len(light)} rows, {len(goals)} with a goal beyond the "
              f"horizon, {len(recs) - n0} scanned", flush=True)
        for p in paths:  # the write pass downloads the few shards it needs again
            os.remove(p)
    df = pd.DataFrame(recs)

    # CLIP: indoor probability and scene type on the current image
    from hnod.environment import IndoorTagger
    tagger = IndoorTagger()
    has = df["thumb"].notna().to_numpy()
    imgs = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for b in df.loc[has, "thumb"]]
    p_in = np.full(len(df), np.nan)
    p_in[has] = tagger(imgs)
    df["indoor_prob"] = p_in
    for name, scenes in (("indoor_scene", INDOOR_SCENES), ("outdoor_scene", OUTDOOR_SCENES)):
        probs = clip_classes(tagger, imgs, list(scenes.values()))
        col = np.full(len(df), None, object)
        col[has] = [list(scenes)[i] for i in probs.argmax(1)]
        df[name] = col
    tag = "_".join(args.sources) if args.sources else "all"
    df.to_parquet(work / f"candidates_{tag}.parquet")
    print("candidates:", len(df), flush=True)


def clip_classes(tagger, images, texts):
    torch = tagger.torch
    with torch.no_grad():
        t = tagger.proc(text=texts, return_tensors="pt", padding=True).to(tagger.device)
        f = tagger.model.get_text_features(**t)
        f = getattr(f, "pooler_output", f)
        text = f / f.norm(dim=-1, keepdim=True)
        out = []
        for i in range(0, len(images), 256):
            x = tagger.proc(images=images[i:i + 256], return_tensors="pt")["pixel_values"].to(tagger.device,
                                                                                            torch.float16)
            g = tagger.model.get_image_features(pixel_values=x)
            g = getattr(g, "pooler_output", g)
            g = g / g.norm(dim=-1, keepdim=True)
            out.append((g @ text.T).float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


MAX_UNKNOWN = 0.3         # reference path at most this unobserved by the lidar map
MIN_GOAL_M = 2.0          # the goal must be at least this far away in a straight line (a reversing or looping
                          # reference can put the "goal beyond the horizon" right next to the start)
MIN_GAP_S = 5.0           # scenarios from one recording at least this far apart in time
MAX_PER_RECORDING = 15    # real recordings (indoor ones are few: a handful of sequences hold them all)
MAX_PER_HOUSE = 4         # simulator houses
MAX_HSSD_INDOOR = 0.75    # share of indoor scenarios that may be simulated (real indoor data is scarce in v1)
OUTDOOR_AT = 0.3          # CLIP indoor probability at or below which a real scenario is outdoor (else reviewed)
SIM_INDOOR_AT = 0.95      # simulator renders: gardens around the houses score 0.3-0.5, rooms about 0.999
SIM_BLACK_SKY = 0.15      # ... and exteriors show the black background where a room would have a ceiling
REAL_FIRST = 10.0         # real scenarios are taken before simulated ones whenever the constraints allow
DUPLICATE_SIM = 0.94      # CLIP image-embedding cosine above which two scenarios count as the same view
AUDIT = Path(__file__).resolve().parents[1] / "docs" / "eval_audit_{version}.json"


def black_top(jpeg, frac=0.4, level=12):
    """Share of near-black pixels in the top of an image (no ceiling: a simulated exterior)."""
    if jpeg is None:
        return 1.0
    im = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return float((im[:int(im.shape[0] * frac)].max(2) < level).mean())


def dataset_of(source_name):
    return source_name.split("_")[0]


def candidate_table(df, audit):
    """Valid candidates with their environment, scene, tags and difficulty.

    Environment: real data uses the reviewed labels of the audit file (every candidate CLIP did not call
    clearly outdoor was looked at); simulated scenarios must look indoor to CLIP with high confidence.
    """
    reviewed, dropped = audit.get("environment", {}), audit.get("drop", {})
    dropped_recordings = audit.get("drop_recordings", {})
    df = df.assign(dataset=df["source"].map(dataset_of))
    env = []
    for r in df.itertuples():
        if r.scenario_id in dropped or f"{r.dataset}/{r.sequence}" in dropped_recordings:
            env.append(None)
        elif r.dataset == "hssd":
            env.append("indoor" if r.indoor_prob >= SIM_INDOOR_AT and black_top(r.thumb) <= SIM_BLACK_SKY else None)
        elif r.scenario_id in reviewed:
            env.append(reviewed[r.scenario_id])
        else:
            env.append("outdoor" if r.indoor_prob <= OUTDOOR_AT else None)  # unreviewed and not clearly outdoor
    df = df.assign(environment=env)
    ok = (df["environment"].notna() & ~df["ref_collided"] & ~df["ref_collided_lidar"] & ~df["start_overlap"]
          & (df["ref_unknown"] <= MAX_UNKNOWN) & ~df["stationary_collided"] & df["thumb"].notna()
          & (df["goal"].map(lambda g: float(np.hypot(*g))) >= MIN_GOAL_M)
          & ~df.get("ref_protocol_collided", False) & ~df.get("ref_reverses", False))
    df = df[ok].copy()
    df["scene"] = np.where(df["environment"] == "indoor", df["indoor_scene"], df["outdoor_scene"])
    df["house"] = np.where(df["dataset"] == "hssd", df["sequence"].str.split("_").str[0], df["sequence"])
    fails = []
    tags = []
    for r in df.itertuples():
        f = [b for b in ("straight_ahead", "straight_to_goal") if not getattr(r, f"{b}_success")]
        fails.append(len(f))
        t = [f"env:{r.environment}", f"source:{r.dataset}", f"scene:{r.scene}", *r.people_tags, *r.path_tags]
        t += [f"baseline_fails:{b}" for b in f]
        if r.luminance < 60:
            t.append("lighting:dim")
        tags.append(sorted(set(t)))
    df["n_baseline_fails"], df["tags"] = fails, tags
    return df


def select(df, n_indoor, n_outdoor, seed=0, emb=None, accepted=()):
    """Greedy: tag coverage first, then difficulty; caps per recording / house / source; no near-duplicate views.

    emb: {scenario_id: unit CLIP image embedding of the current image}.
    accepted: scenarios that passed the audit; they are kept (if still valid) and only the rest is filled.
    """
    rng = np.random.default_rng(seed)
    df = df.assign(rand_=rng.random(len(df)))
    chosen = []
    for env, n in (("indoor", n_indoor), ("outdoor", n_outdoor)):
        pool = df[df["environment"] == env]
        caps = {"hssd": int(round(MAX_HSSD_INDOOR * n))} if env == "indoor" else {}
        counts, per_rec, times, src_n, views = Counter(), Counter(), defaultdict(list), Counter(), defaultdict(list)
        picked = []

        def take(r):
            picked.append(r.scenario_id)
            counts.update(r.tags)
            per_rec[(r.dataset, r.house)] += 1
            src_n[r.dataset] += 1
            times[(r.source, r.sequence, r.segment)].append(r.time_s)
            if emb is not None:
                views[(r.dataset, r.house)].append(emb[r.scenario_id])

        for r in pool[pool["scenario_id"].isin(set(accepted))].itertuples():
            if len(picked) < n:
                take(r)
        while len(picked) < n:
            best, best_score = None, -1.0
            for r in pool.itertuples():
                if r.scenario_id in picked:
                    continue
                rec = (r.dataset, r.house)
                if per_rec[rec] >= (MAX_PER_HOUSE if r.dataset == "hssd" else MAX_PER_RECORDING):
                    continue
                if src_n[r.dataset] >= caps.get(r.dataset, n):
                    continue
                if any(abs(r.time_s - t) < MIN_GAP_S for t in times[(r.source, r.sequence, r.segment)]):
                    continue
                # simulator episodes of one house can start from the same pose; real recordings are kept apart
                # by MIN_GAP_S instead (corridor frames seconds apart look alike but are different situations)
                if emb is not None and r.dataset == "hssd" and views[rec] \
                        and max(float(emb[r.scenario_id] @ v) for v in views[rec]) > DUPLICATE_SIM:
                    continue
                # tag coverage first, then difficulty, then spread over recordings and sources
                cover = sum(1.0 / (1 + counts[t]) for t in r.tags if not t.startswith("source:"))
                sc = cover + 1.5 * r.n_baseline_fails + 0.5 / (1 + per_rec[rec]) + 0.5 / (1 + src_n[r.dataset]) \
                    + 0.1 * r.rand_ + (REAL_FIRST if r.dataset != "hssd" else 0.0)
                if sc > best_score:
                    best, best_score = r, sc
            if best is None:
                print(f"{env}: only {len(picked)} scenarios satisfy the constraints", flush=True)
                break
            take(best)
        chosen += picked
    return df[df["scenario_id"].isin(chosen)].drop(columns=["rand_"])


def cmd_select(args):
    work = Path(args.work)
    audit = json.load(open(str(AUDIT).format(version=args.version)))
    cand = pd.concat([pd.read_parquet(p) for p in sorted(work.glob("candidates_*.parquet"))], ignore_index=True)
    df = candidate_table(cand, audit)
    emb = None
    if (work / "clip_embeddings.npy").exists():
        E = np.load(work / "clip_embeddings.npy")
        ids = pd.read_parquet(work / "clip_embeddings_ids.parquet")["scenario_id"]
        emb = dict(zip(ids, E))
    if emb is not None:  # views that nearly match a rejected scenario of the same house / recording
        cand_all = cand.assign(dataset=cand["source"].map(dataset_of))
        cand_all["house"] = np.where(cand_all["dataset"] == "hssd", cand_all["sequence"].str.split("_").str[0],
                                     cand_all["sequence"])
        rejected = cand_all[cand_all["scenario_id"].isin(audit.get("drop", {}))]
        bad = set()
        for r in rejected.itertuples():
            if r.scenario_id not in emb:
                continue
            same = df[(df["dataset"] == r.dataset) & (df["house"] == r.house)]
            bad |= {x for x in same["scenario_id"] if x in emb and float(emb[x] @ emb[r.scenario_id]) > DUPLICATE_SIM}
        df = df[~df["scenario_id"].isin(bad - set(audit.get("accepted", [])))]
    print("valid candidates:", len(df), dict(Counter(zip(df["environment"], df["dataset"]))), flush=True)
    sel = select(df, args.indoor, args.outdoor, args.seed, emb, audit.get("accepted", []))
    sel.drop(columns=["thumb"]).to_parquet(work / f"selected_{args.version}.parquet")
    print("selected:", len(sel), dict(Counter(zip(sel["environment"], sel["dataset"]))))
    tags = Counter(t for ts in sel["tags"] for t in ts)
    for t, n in sorted(tags.items()):
        print(f"  {t}: {n}")


PROMPT = "Please walk towards the goal ({x:.1f}, {y:.1f})."


def eval_features():
    from datasets import Features, List, Value
    from hnod.io import FEATURES
    extra = {"suite_id": Value("string"), "suite_version": Value("string"), "prompt": Value("string"),
             "environment": Value("string"), "scene": Value("string"), "tags": List(Value("string")),
             "source": Value("string"), "source_repo": Value("string"), "source_config": Value("string"),
             "source_split": Value("string"), "reference": Value("string"), "goal_extra_m": Value("float32")}
    return Features({**extra, **FEATURES})


def render_audit(row, path):
    """Current image with the recorded path, and the bird's-eye view with goal and baselines."""
    import io as _io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from visualize import draw, draw_image
    r = dict(row)
    r["past_images"] = [PILImage.open(_io.BytesIO(im["bytes"])).convert("RGB") for im in row["past_images"]]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    try:
        draw_image(ax[0], r)
    except Exception:  # cameras without a usable pose: show the image only
        ax[0].imshow(r["past_images"][-1])
        ax[0].axis("off")
    draw(ax[1], r, view=float(np.clip(np.hypot(*row["goal"]) + 1.5, 8, 16)))
    for name, p in suite.baseline_paths(r).items():
        steps = suite.timed_positions(p, r["rate_hz"])
        ax[1].plot(steps[:, 1], steps[:, 0], "--", lw=1, label=name)
    ax[1].legend(fontsize=7, loc="lower right")
    ax[0].set_title(f"{row['suite_id']}  {row['prompt']}", fontsize=9)
    ax[1].set_title(" ".join(t for t in row["tags"] if not t.startswith("source:")), fontsize=7, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def cmd_write(args):
    from datasets import Dataset
    work = Path(args.work)
    sel = pd.read_parquet(work / f"selected_{args.version}.parquet")
    src = {s["name"]: s for s in SOURCES}
    sel = sel.sort_values(["environment", "source", "scenario_id"], ascending=[True, True, True])
    sel["suite_id"] = [f"{args.version}-{i:04d}" for i in range(len(sel))]
    info = sel.set_index("scenario_id")
    rows = []
    groups = list(sel.groupby("shard"))

    def fetch(item):
        shard, grp = item
        s = src[grp["source"].iloc[0]]
        rel = shard.split(f"/src/{s['name']}/", 1)[1]
        return hf_hub_download(s["repo"], rel, repo_type="dataset", local_dir=work / "src" / s["name"])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:  # shards stay on disk: rewrites after an audit are then quick
        locals_ = list(pool.map(fetch, groups))
    for (shard, grp), local in zip(groups, locals_):
        s = src[grp["source"].iloc[0]]
        want = set(grp["scenario_id"])
        for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if row["scenario_id"] not in want:
                    continue
                m = info.loc[row["scenario_id"]]
                goal = [float(v) for v in m["goal"]]
                row.update(suite_id=m["suite_id"], suite_version=args.version,
                           prompt=PROMPT.format(x=goal[0], y=goal[1]), environment=m["environment"],
                           scene=m["scene"], tags=list(m["tags"]), source=dataset_of(s["name"]), source_repo=s["repo"],
                           source_config=s["config"], source_split=s["split"], reference=s["reference"],
                           goal_extra_m=float(m["goal_extra_m"]), goal=goal, task="pointgoal", instruction="")
                row.setdefault("task", "pointgoal")
                rows.append(row)
    rows.sort(key=lambda r: r["suite_id"])
    out = work / "suite" / args.version
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    feats = eval_features()
    per = 40
    n = (len(rows) + per - 1) // per
    for k in range(n):
        Dataset.from_list(rows[k * per:(k + 1) * per], features=feats).to_parquet(
            out / "data" / f"test-{k:05d}-of-{n:05d}.parquet")
    for r in rows:
        render_audit(r, out / "audit" / f"{r['suite_id']}.png")
    sel.drop(columns=["shard"]).to_parquet(out / "selection.parquet")
    print(f"wrote {len(rows)} scenarios to {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["candidates", "select", "write"])
    ap.add_argument("--work", default="/workspace/cache/suite")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--sources", nargs="*", help="candidates: only these sources")
    ap.add_argument("--indoor", type=int, default=100)
    ap.add_argument("--outdoor", type=int, default=50)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "select": cmd_select, "write": cmd_write}[a.command](a)
