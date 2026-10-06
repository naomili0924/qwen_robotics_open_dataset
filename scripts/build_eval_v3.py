#!/usr/bin/env python
"""Evaluation suite v3: motion-diverse real scenarios (the v2 set was two thirds "keep walking straight").

Candidates come from build_eval_suite.py's pass 1 (held-out real recordings: CODa, JRDB, RoboSense, MuSoHu) with the
v2 audit's environment labels and drops.  Each candidate's recorded path over the final-frame horizon is classed with
the training set's motion rule (scripts/build_dedup_index.py: standing, stopping, starting, sharp / gentle turns,
slowing, speeding up, weaving, straight) and the selection fills a quota per class, preferring scenarios with people
and ones where the naive baselines fail, at most a few per recording and spaced in time.

    python scripts/build_eval_v3.py features      # -> <work>/motion_v3.parquet (reads the held-out shards)
    python scripts/build_eval_v3.py select        # -> <work>/selected_v3.parquet, then build_eval_suite.py write --version v3
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dedup_index import bucket  # noqa: E402
from build_eval_suite import AUDIT, MAX_PER_RECORDING, MAX_UNKNOWN, MIN_GAP_S, OUTDOOR_AT, dataset_of  # noqa: E402
from hnod.scenario import CURRENT  # noqa: E402

HORIZON_S = 5.0
# scenarios per motion class (about 150 in all); classes with fewer candidates give what they have
QUOTA = {"sharp_left": 15, "sharp_right": 15, "turn_left": 15, "turn_right": 15, "stopping": 12, "starting": 8,
         "standing": 8, "slowing": 12, "speeding_up": 12, "weaving": 10, "straight": 30}
MIN_INDOOR_SHARE = 0.6
CLIP_INDOOR_AT = 0.7  # unreviewed scenarios: CLIP indoor probability at or above which they count as indoor


def motion_of(ego, rate_hz):
    """Training-set motion features of the recorded path over the next HORIZON_S seconds (10 positions, by time)."""
    x, y, h = (np.asarray(ego[k], float) for k in ("x", "y", "heading"))
    t = np.arange(len(x)) / rate_hz
    t0 = t[CURRENT]
    horizon = min(HORIZON_S, t[-1] - t0)  # the row's future may be shorter than the horizon (JRDB: 4 s)
    times = t0 + horizon * np.arange(1, 11) / 10
    xy = np.c_[np.interp(times, t, x), np.interp(times, t, y)] - [x[CURRENT], y[CURRENT]]
    c, s = np.cos(h[CURRENT]), np.sin(h[CURRENT])
    loc = np.c_[c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]]
    yaw = np.interp(times, t, np.unwrap(h)) - h[CURRENT]
    seg = np.hypot(*np.diff(np.vstack([[0, 0], loc]), axis=0).T)
    speed = seg / (horizon / 10)
    bearing = np.degrees(np.arctan2(loc[-1, 1], loc[-1, 0]))
    return dict(dist_m=float(seg.sum()), speed_mean=float(speed.mean()), speed_first=float(speed[:2].mean()),
                speed_last=float(speed[-2:].mean()), speed_min=float(speed.min()), heading_deg=float(np.degrees(yaw[-1])),
                bearing_deg=float(bearing), lateral_m=float(loc[-1, 1]),
                path_straightness=float(seg.sum() / max(1e-6, np.hypot(*loc[-1]))))


def load_candidates(work):
    return pd.concat([pd.read_parquet(p) for p in sorted(work.glob("candidates_*.parquet"))], ignore_index=True)


def cmd_features(args):
    work = Path(args.work)
    cand = load_candidates(work)
    cand = cand[~cand["source"].str.startswith("hssd")]
    rows = []
    for shard, grp in cand.groupby("shard"):
        want = set(grp["scenario_id"])
        t = pq.read_table(shard, columns=["scenario_id", "rate_hz", "ego"]).to_pylist()
        for r in t:
            if r["scenario_id"] in want:
                f = motion_of(r["ego"], r["rate_hz"])
                rows.append(dict(scenario_id=r["scenario_id"], bucket=bucket(f), **f))
        print(f"{Path(shard).parent.parent.parent.name} {Path(shard).stem}: {len(want)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(work / "motion_v3.parquet")
    print(df["bucket"].value_counts().to_string())


def valid_table(cand, audit):
    """Like build_eval_suite.candidate_table, with the validity rules of timed scoring.

    Kept: the audit's drops and reviewed environment labels; the recorded reference must be collision-free, not
    overlapping an obstacle at the start and not reversing.  Relaxed: the share of the reference over lidar-unobserved
    map cells (MAX_UNKNOWN) is not required - it removed most turns, stops and all standing scenarios, because the
    cells right around the sensor and to its sides are unobserved; such scenarios are tagged `map:partly_unknown`
    (their map-collision score is less reliable, their trajectory scores are not).  The 0.5 m/s protocol collision
    and the stationary-baseline collision are irrelevant to timed scoring.  Unreviewed scenarios take CLIP's label
    when it is confident (tag `env:by_clip`); the goal-distance rule is dropped (the final frame is the goal).
    """
    reviewed, dropped, dropped_recordings = audit.get("environment", {}), audit.get("drop", {}), audit.get("drop_recordings", {})
    df = cand.assign(dataset=cand["source"].map(dataset_of))
    df = df[df["dataset"] != "hssd"]
    env, by_clip = [], []
    for r in df.itertuples():
        if r.scenario_id in dropped or f"{r.dataset}/{r.sequence}" in dropped_recordings:
            env.append(None); by_clip.append(False)
        elif r.scenario_id in reviewed:
            env.append(reviewed[r.scenario_id]); by_clip.append(False)
        elif r.indoor_prob <= OUTDOOR_AT:
            env.append("outdoor"); by_clip.append(False)
        elif r.indoor_prob >= CLIP_INDOOR_AT:
            env.append("indoor"); by_clip.append(True)
        else:
            env.append(None); by_clip.append(False)
    df = df.assign(environment=env, env_by_clip=by_clip)
    ok = (df["environment"].notna() & ~df["ref_collided"] & ~df["ref_collided_lidar"] & ~df["start_overlap"]
          & ~df["ref_reverses"].fillna(False) & df["thumb"].notna())
    df = df[ok].copy()
    df["scene"] = np.where(df["environment"] == "indoor", df["indoor_scene"], df["outdoor_scene"])
    df["house"] = df["sequence"]
    fails, tags = [], []
    for r in df.itertuples():
        f = [b for b in ("straight_ahead", "straight_to_goal") if not getattr(r, f"{b}_success")]
        fails.append(len(f))
        t = [f"env:{r.environment}", f"source:{r.dataset}", f"scene:{r.scene}", *r.people_tags, *r.path_tags]
        t += [f"baseline_fails:{b}" for b in f]
        if r.luminance < 60:
            t.append("lighting:dim")
        if r.ref_unknown > MAX_UNKNOWN:
            t.append("map:partly_unknown")
        if r.env_by_clip:
            t.append("env:by_clip")
        tags.append(sorted(set(t)))
    df["n_baseline_fails"], df["tags"] = fails, tags
    return df


def cmd_select(args):
    work = Path(args.work)
    audit = json.load(open(str(AUDIT).format(version=args.based_on)))
    cand = load_candidates(work)
    df = valid_table(cand, audit).merge(pd.read_parquet(work / "motion_v3.parquet"), on="scenario_id")
    print("valid real candidates:", len(df))
    print(pd.crosstab(df["bucket"], [df["environment"], df["dataset"]]).to_string())
    rng = np.random.default_rng(args.seed)
    df = df.assign(rand_=rng.random(len(df)))
    picked, per_rec, times, env_n = [], Counter(), defaultdict(list), Counter()
    total = sum(QUOTA.values())

    def ok(r):
        if per_rec[(r.dataset, r.house)] >= MAX_PER_RECORDING:
            return False
        if any(abs(r.time_s - t) < MIN_GAP_S for t in times[(r.source, r.sequence, r.segment)]):
            return False
        # keep the indoor majority: outdoor may not exceed its share of the target size
        return not (r.environment == "outdoor" and env_n["outdoor"] + 1 > (1 - MIN_INDOOR_SHARE) * total)

    def take(r):
        picked.append(r.scenario_id)
        per_rec[(r.dataset, r.house)] += 1
        times[(r.source, r.sequence, r.segment)].append(r.time_s)
        env_n[r.environment] += 1

    # rare classes first so that the recording caps do not fill up with straight segments
    for b, n in sorted(QUOTA.items(), key=lambda kv: kv[1]):
        pool = df[df["bucket"] == b]
        got = 0
        while got < n:
            best, best_score = None, -1.0
            for r in pool.itertuples():
                if r.scenario_id in picked or not ok(r):
                    continue
                people = 0 if "people:none" in r.tags else 1
                sc = 2.0 * r.n_baseline_fails + 1.5 * people + (1.0 if r.environment == "indoor" else 0.0) \
                    + 0.5 / (1 + per_rec[(r.dataset, r.house)]) + 0.3 * r.rand_
                if sc > best_score:
                    best, best_score = r, sc
            if best is None:
                break
            take(best)
            got += 1
        print(f"{b}: {got} of {n} (candidates {len(pool)})", flush=True)
    sel = df[df["scenario_id"].isin(picked)].drop(columns=["rand_"])
    sel["tags"] = [sorted(set(t) | {f"motion:{b}"}) for t, b in zip(sel["tags"], sel["bucket"])]
    sel.drop(columns=["thumb"]).to_parquet(work / f"selected_{args.version}.parquet")
    print("selected:", len(sel), dict(Counter(zip(sel["environment"], sel["dataset"]))))
    print(sel["bucket"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "select"])
    ap.add_argument("--work", default="/workspace/cache/suite")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--based-on", default="v2", help="audit file whose environment labels and drops are reused")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"features": cmd_features, "select": cmd_select}[args.command](args)


if __name__ == "__main__":
    main()
