"""Cut training samples from per-frame data (hnod.frames) at load time.

A sample is anchored at one frame (the current moment) of one episode:

* history: ``n_past`` earlier frames every ``past_dt_s`` seconds plus the current one (repeated
  at the episode start, with a mask), and the robot's poses at those times;
* target: ``n_waypoints`` poses (x, y, yaw) taken every ``spacing_m`` metres *along the path*, so
  a source's walking speed is not baked in; past the end of an episode that ends at rest the
  last pose repeats and ``stop`` is set;
* goal: a point further along the path than the target horizon (``goal_min_m`` .. ``goal_max_m``),
  or the end of a language segment, whose instruction then names it;
* prompt: the task in words; the goal is part of the prompt.

Coordinates are metres in the robot frame at the current moment: robot at (0, 0), x forward,
y left; yaw relative to the current heading.
"""
from dataclasses import dataclass

import numpy as np

POINT_TEMPLATES = ["Please walk towards the goal ({x:.1f}, {y:.1f}).",
                   "Walk to the point ({x:.1f}, {y:.1f}).",
                   "Navigate to ({x:.1f}, {y:.1f})."]
LANGUAGE_TEMPLATES = ["Walk to {what}.", "Go to {what}.", "Head towards {what}."]
LANGUAGE_POINT_SUFFIX = " It is at ({x:.1f}, {y:.1f})."


@dataclass
class WindowConfig:
    n_waypoints: int = 8
    spacing_m: float = 0.25
    n_past: int = 10
    past_dt_s: float = 0.5
    goal_min_m: float = 4.0
    goal_max_m: float = 20.0
    language_fraction: float = 0.5   # of the samples inside a language segment, the share that uses it
    language_point_fraction: float = 0.5  # of language prompts, the share that also gives the point
    min_future_m: float = 0.5        # samples whose remaining path is shorter are skipped (unless at rest)
    stride: int = 1                  # every stride-th frame is a sample
    still_m: float = 0.05            # movement below this (odometry jitter) does not count as travel
    min_indoor_prob: float = 0.0     # keep only samples whose current frame is at least this likely indoor
    seed: int = 0
    # mode "final_frame": the model sees past frames and the one frame `horizon_s` ahead and must reproduce
    # the recorded motion in between: n_waypoints positions evenly spaced IN TIME (speed is part of the target).
    # No goal coordinates, no instruction; the prompt only names the embodiment.
    mode: str = "prompt"             # prompt (goal in words / coordinates, waypoints by distance) | final_frame
    horizon_s: float = 5.0
    camera_prompt: bool = False  # final_frame: add the camera sentence (field of view, height or 'unknown') to the prompt


EMBODIMENT_TAGS = {"person_walking": "a human walking", "wheeled_robot": "a robot", "legged_robot": "a robot",
                   "simulated_agent": "a robot"}
# In the style of Qwen-VLA's embodiment-aware prompt (arXiv 2605.30280, section 2.3): who carries the camera, the
# rate of the observations and the rate and number of predicted positions.  No task or instruction at this stage.
EMBODIMENT_TEMPLATE = ("The camera is carried by {tag}. The past views are sampled at {past_hz:g} Hz. "
                       "Please predict the next {n} positions at {out_hz:g} Hz.")


KNOWN_HEIGHT_SOURCES = ("dataset", "simulator")  # calibration we are certain of; anything else is "unknown"


def camera_prompt(camera):
    """One sentence about the camera, from stored calibration only: the field of view from K and the height above
    the ground when its source is the dataset's calibration or a simulator.  Nothing is estimated or guessed: a
    missing or estimated value is stated as unknown, so the model knows it has to infer it from the images."""
    cam = camera or {}
    K, w, h = cam.get("K"), cam.get("width"), cam.get("height")
    if K is not None and len(K) >= 5 and w and K[0] and K[0] > 0:
        hfov = 2 * np.degrees(np.arctan(w / (2 * K[0])))
        text = f"The camera's horizontal field of view is {hfov:.0f} degrees"
        if h and K[4] and K[4] > 0:
            text += f" and its vertical field of view {2 * np.degrees(np.arctan(h / (2 * K[4]))):.0f} degrees"
    else:
        text = "The camera's field of view is unknown"
    hm = cam.get("height_m")
    if hm is not None and np.isfinite(hm) and cam.get("height_source") in KNOWN_HEIGHT_SOURCES:
        return text + f"; it is mounted {hm:.2f} m above the ground."
    return text + "; its height above the ground is unknown."


def embodiment_prompt(embodiment, past_dt_s=1.0, n=10, horizon_s=5.0, camera=None):
    """The embodiment sentence; with `camera` (an episode's camera dict, possibly empty) also the camera sentence."""
    text = EMBODIMENT_TEMPLATE.format(tag=EMBODIMENT_TAGS.get(embodiment, "a robot"), past_hz=1.0 / past_dt_s, n=n,
                                      out_hz=n / horizon_s)
    return text if camera is None else text + " " + camera_prompt(camera)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_frame(pose0, xy, yaw=None):
    """World (x, y[, yaw]) -> frame of pose0 = (x, y, z, yaw)."""
    c, s = np.cos(pose0[3]), np.sin(pose0[3])
    d = np.asarray(xy, float) - pose0[:2]
    local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)
    if yaw is None:
        return local
    return local, _wrap(np.asarray(yaw) - pose0[3])


def _lower_first(text):
    """'A white door.' -> 'a white door'; acronyms ('TV stand') keep their case."""
    text = text.strip().rstrip(".")
    first = text.split()[0] if text else ""
    if len(first) > 1 and first.isupper():
        return text
    return text[:1].lower() + text[1:]


def arc_length(xy, still_m=0.05):
    """Distance travelled along a path, ignoring jitter: a point counts only once it is more than
    ``still_m`` from the last counted point (odometry noise while standing would otherwise add up)."""
    s = np.zeros(len(xy))
    anchor, total = xy[0], 0.0
    for j in range(1, len(xy)):
        d = float(np.hypot(*(xy[j] - anchor)))
        if d > still_m:
            total += d
            anchor = xy[j]
        s[j] = total
    return s


class FrameWindows:
    """Index of samples over a ``frames`` split; images are read lazily.

    frames: datasets.Dataset of the frames config (any columns; only ``image`` is read per item).
    episodes: datasets.Dataset (or list of dicts) of the episodes config for the same split.
    """

    def __init__(self, frames, episodes, cfg=None, allowed=None):
        """allowed: optional set of (episode_id, frame_index) to restrict the samples to (a published sample list)."""
        self.frames, self.cfg, self.allowed = frames, cfg or WindowConfig(), allowed
        table = frames.data
        ep_col = np.asarray(table.column("episode_id").to_pylist())
        self.t = table.column("timestamp").to_numpy()
        self.pose = np.asarray(table.column("pose").to_pylist(), float)
        self.episodes = {e["episode_id"]: e for e in episodes}
        change = np.r_[True, ep_col[1:] != ep_col[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(ep_col)]
        assert len(set(ep_col[starts])) == len(starts), "rows of an episode must be contiguous"
        self.ep_of_row = np.repeat(np.arange(len(starts)), ends - starts)
        self.ep_ids, self.ep_start, self.ep_end = ep_col[starts], starts, ends
        s = np.zeros(len(ep_col))
        for a, b in zip(starts, ends):  # strictly increasing for interp
            s[a:b] = arc_length(self.pose[a:b, :2], self.cfg.still_m) + 1e-9 * np.arange(b - a)
        self.s = s
        self.samples = self._eligible()

    def _eligible(self):
        c, out = self.cfg, []
        for e, (a, b) in enumerate(zip(self.ep_start, self.ep_end)):
            at_rest = bool(self.episodes.get(self.ep_ids[e], {}).get("ends_at_rest", False))
            need = c.min_future_m if at_rest else c.n_waypoints * c.spacing_m
            rows = np.arange(a, b, c.stride)
            left = self.s[b - 1] - self.s[rows]
            keep = (left >= need) | (at_rest & (left >= 0))
            if c.mode == "final_frame":  # the recording must continue for the whole horizon (standing still counts)
                keep = self.t[b - 1] - self.t[rows] >= c.horizon_s - 1e-6
            prob = self.episodes.get(self.ep_ids[e], {}).get("frame_indoor_prob")
            if c.min_indoor_prob > 0 and prob is not None and len(prob) == b - a:
                keep &= np.asarray(prob)[rows - a] >= c.min_indoor_prob
            if self.allowed is not None:
                eid = str(self.ep_ids[e])
                keep &= np.array([(eid, int(r - a)) in self.allowed for r in rows], bool)
            out.append(rows[keep])
        return np.concatenate(out) if out else np.zeros(0, int)

    def __len__(self):
        return len(self.samples)

    def _pose_at_s(self, a, b, s):
        """Interpolated (x, y, yaw) at arc lengths s within rows a..b-1."""
        ss = self.s[a:b]
        x = np.interp(s, ss, self.pose[a:b, 0])
        y = np.interp(s, ss, self.pose[a:b, 1])
        yaw = np.interp(s, ss, np.unwrap(self.pose[a:b, 3]))
        return np.stack([x, y], -1), yaw

    def sample(self, k):
        """Everything but the images: rows of the history frames, poses, goal, prompt."""
        c, i = self.cfg, int(self.samples[k])
        e = self.ep_of_row[i]
        a, b = self.ep_start[e], self.ep_end[e]
        ep = self.episodes.get(self.ep_ids[e], {})
        rng = np.random.default_rng([c.seed, i])
        p0 = self.pose[i]
        if c.mode == "final_frame":
            return self._final_frame_sample(i, e, a, b, ep)

        # history: frames at t - j * past_dt, nearest at or before, clamped to the episode start
        want = self.t[i] - c.past_dt_s * np.arange(c.n_past, -1, -1)
        rows = a + np.searchsorted(self.t[a:b], want + 1e-6, side="right") - 1
        past_mask = rows >= a
        rows = np.clip(rows, a, i)
        past_xy = to_frame(p0, self.pose[rows, :2])
        prev = max(i - 1, a)
        dt = max(self.t[i] - self.t[prev], 1e-3)
        v = (self.pose[i, :2] - self.pose[prev, :2]) / dt
        vel = to_frame(np.r_[0.0, 0.0, 0.0, p0[3]], v)  # rotate only

        # target waypoints by distance along the path
        s_end = self.s[b - 1]
        s_want = self.s[i] + c.spacing_m * np.arange(1, c.n_waypoints + 1)
        stop = s_want > s_end
        xy, yaw = self._pose_at_s(a, b, np.minimum(s_want, s_end))
        wxy, wyaw = to_frame(p0, xy, yaw)
        target = np.c_[wxy, wyaw].astype(np.float32)

        # goal: a language segment that covers this frame, or a point beyond the horizon
        fi = i - a
        horizon = c.n_waypoints * c.spacing_m  # a language goal must still be ahead of the scored horizon
        segs = [g for g in ep.get("segments") or [] if g["start_frame"] <= fi < g["end_frame"]
                and self.s[a + g["end_frame"]] - self.s[i] >= horizon]
        use_lang = bool(segs) and rng.random() < c.language_fraction
        if use_lang:
            seg = segs[rng.integers(len(segs))]
            gxy = to_frame(p0, self.pose[a + seg["end_frame"], :2])
            what = _lower_first(seg["instruction"])
            prompt = LANGUAGE_TEMPLATES[rng.integers(len(LANGUAGE_TEMPLATES))].format(what=what)
            give_point = rng.random() < c.language_point_fraction
            if give_point:
                prompt += LANGUAGE_POINT_SUFFIX.format(x=gxy[0], y=gxy[1])
            task, instruction = "language", seg["instruction"]
        else:
            d = rng.uniform(c.goal_min_m, c.goal_max_m)
            g, _ = self._pose_at_s(a, b, np.array([min(self.s[i] + d, s_end)]))
            gxy = to_frame(p0, g[0])
            prompt = POINT_TEMPLATES[rng.integers(len(POINT_TEMPLATES))].format(x=gxy[0], y=gxy[1])
            task, instruction, give_point = "pointgoal", "", True
        return dict(row=i, episode_id=str(self.ep_ids[e]), frame_index=int(fi), history_rows=rows,
                    history_mask=past_mask, past_xy=past_xy.astype(np.float32), velocity=vel.astype(np.float32),
                    target=target, stop=stop, goal=gxy.astype(np.float32), goal_given=give_point,
                    task=task, instruction=instruction, prompt=prompt,
                    dataset=ep.get("dataset", ""), rate_hz=float(ep.get("rate_hz", 0) or 0))

    def _final_frame_sample(self, i, e, a, b, ep):
        """Past frames + the frame horizon_s ahead -> the recorded positions in between, by time."""
        c, p0, t0 = self.cfg, self.pose[i], self.t[i]
        tt = self.t[a:b]
        want = t0 - c.past_dt_s * np.arange(c.n_past, -1, -1)
        # the frame nearest each past time (the same rule as scripts/publish_dedup.py, which stores images only for
        # these frames), never after the current one
        rows = np.minimum(a + np.abs(tt[:, None] - want[None]).argmin(0), i)
        times = t0 + c.horizon_s * np.arange(1, c.n_waypoints + 1) / c.n_waypoints
        xy = np.stack([np.interp(times, tt, self.pose[a:b, 0]), np.interp(times, tt, self.pose[a:b, 1])], -1)
        yaw = np.interp(times, tt, np.unwrap(self.pose[a:b, 3]))
        wxy, wyaw = to_frame(p0, xy, yaw)
        j = a + int(np.argmin(np.abs(tt - (t0 + c.horizon_s))))  # the frame nearest the end of the horizon
        embodiment = ep.get("embodiment", "") or ""
        return dict(row=i, episode_id=str(self.ep_ids[e]), frame_index=int(i - a), history_rows=rows,
                    final_row=j, final_dt_s=float(self.t[j] - t0), target=np.c_[wxy, wyaw].astype(np.float32),
                    embodiment=embodiment, task="final_frame",
                    prompt=embodiment_prompt(embodiment, c.past_dt_s, c.n_waypoints, c.horizon_s,
                                             camera=(ep.get("camera") or {}) if c.camera_prompt else None),
                    dataset=ep.get("dataset", ""), rate_hz=float(ep.get("rate_hz", 0) or 0))

    def images(self, rows, column="image"):
        """PIL images for frame rows (repeats allowed)."""
        uniq, inv = np.unique(rows, return_inverse=True)
        got = self.frames.select(uniq.tolist()).select_columns([column])[column]
        if any(g is None for g in got):
            raise ValueError(f"no image stored for rows {uniq[[g is None for g in got]].tolist()} (a reduced repo "
                             "stores images only for the frames its listed samples need)")
        return [got[j] for j in inv]

    def __getitem__(self, k):
        out = self.sample(k)
        if "final_row" in out:  # history (oldest first, the last is now) then the final frame
            imgs = self.images(np.r_[out["history_rows"], out["final_row"]])
            out["images"], out["final_image"] = imgs[:-1], imgs[-1]
        else:
            out["images"] = self.images(out["history_rows"])
        return out
