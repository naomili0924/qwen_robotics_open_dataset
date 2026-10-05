"""Per-frame storage format for training data (one Hugging Face dataset repo per source).

Two configs per repo, both split train / validation / test by recording:

* ``frames``: one row per camera frame, rows of an episode contiguous and in time order.
  ``pose`` is the raw metric pose (x, y, z, yaw) in the episode's own odometry frame, z up;
  nothing is resampled or normalised, so any speed, horizon or waypoint spacing can be cut at
  load time (``hnod.windows``).
* ``episodes``: one row per episode: camera, embodiment, environment, language segments.

An episode is a stretch with valid, continuous poses; sources are cut at odometry failures.
Windows, waypoints, goals and prompts are made at load time, never stored.
"""
import cv2
import numpy as np
from datasets import Dataset, Features, Image, List, Value

FRAME_FEATURES = Features({
    "episode_id": Value("string"),
    "frame_index": Value("int32"),     # 0-based within the episode
    "source_frame": Value("int32"),    # frame number in the source recording
    "timestamp": Value("float64"),     # seconds since the episode's first frame
    "image": Image(),                  # front camera (left camera of a stereo pair), native resolution
    "image_right": Image(),            # right camera of a stereo pair, None if the source has none
    "pose": List(Value("float64")),    # x, y, z (m), yaw (rad) in the episode odometry frame, z up
})

SEGMENT = {"start_frame": Value("int32"), "end_frame": Value("int32"), "task": Value("string"),
           "instruction": Value("string"), "brief": Value("string"), "source": Value("string")}

EPISODE_FEATURES = Features({
    "episode_id": Value("string"),
    "dataset": Value("string"),
    "source_sequence": Value("string"),
    "split": Value("string"),
    "num_frames": Value("int32"),
    "rate_hz": Value("float32"),
    "duration_s": Value("float32"),
    "path_length_m": Value("float32"),
    "median_speed_mps": Value("float32"),
    # person_walking | wheeled_robot | legged_robot | simulated_agent | car
    "embodiment": Value("string"),
    # indoor | outdoor | mixed | unknown, and how it was decided (labelled | estimated:<method>)
    "environment": Value("string"),
    "environment_method": Value("string"),
    "indoor_fraction": Value("float32"),
    "frame_indoor_prob": List(Value("float32")),  # per frame, same method; lets loaders pick indoor windows
    "camera": {"name": Value("string"), "width": Value("int32"), "height": Value("int32"),
               "K": List(Value("float64")),           # fx, 0, cx, 0, fy, cy, 0, 0, 1
               "distortion_model": Value("string"),   # opencv_rational | opencv | fisheye | none
               "distortion": List(Value("float64")),
               "height_m": Value("float32"),           # above the ground, NaN if unknown
               "stereo_baseline_m": Value("float32")},  # NaN if mono
    # language goals: frames start..end (inclusive), the instruction names what is reached at end_frame
    "segments": List(SEGMENT),
    # True if the episode ends because the walker stopped (a real "stop here" example); False if it
    # ends at a recording cut or odometry failure, so its last frames must not teach stopping
    "ends_at_rest": Value("bool"),
    "fits": Value("string"),           # both | robotnav (see docs/model_formats.md)
    "licence": Value("string"),
})

EMBODIMENTS = {"person_walking", "wheeled_robot", "legged_robot", "simulated_agent", "car"}


def yaw_from_quat(qx, qy, qz, qw):
    """Heading of the body x axis in the world x-y plane (z up)."""
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def cut_episodes(t, xy, valid, max_step=1.0, max_gap_s=1.0, min_frames=20, min_length_m=3.0):
    """[(start, end)) index ranges of stretches with valid, continuous poses.

    Cut at invalid poses, position jumps over ``max_step`` metres between consecutive frames
    (odometry re-initialisations) and time gaps over ``max_gap_s``; keep stretches with at least
    ``min_frames`` frames and ``min_length_m`` of travel.
    """
    t, xy, valid = np.asarray(t, float), np.asarray(xy, float), np.asarray(valid, bool)
    valid = valid & np.isfinite(xy).all(1)
    brk = np.zeros(len(t), bool)  # brk[i]: frame i cannot follow frame i-1
    step = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
    gap = np.r_[0.0, np.diff(t)]
    brk[1:] = (step[1:] > max_step) | (gap[1:] > max_gap_s) | ~np.isfinite(step[1:])
    out, start = [], None
    for i in range(len(t) + 1):
        if i == len(t) or not valid[i] or (brk[i] and start is not None):
            if start is not None:
                seg = xy[start:i]
                if i - start >= min_frames and np.hypot(*np.diff(seg, axis=0).T).sum() >= min_length_m:
                    out.append((start, i))
            start = i if i < len(t) and valid[i] else None
        elif start is None:
            start = i
    return out


def path_stats(t, xy):
    d = np.hypot(*np.diff(xy, axis=0).T)
    dt = np.diff(t)
    speed = d / np.maximum(dt, 1e-6)
    return float(d.sum()), float(np.median(speed)) if len(speed) else 0.0


def jpeg_bytes(img_bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def write_frames(rows, path):
    """rows: dicts matching FRAME_FEATURES (images as {"bytes", "path"}); written in the given order."""
    Dataset.from_list(rows, features=FRAME_FEATURES).to_parquet(path)


def write_episodes(rows, path):
    Dataset.from_list(rows, features=EPISODE_FEATURES).to_parquet(path)


def split_of(key, val=0.05, test=0.05):
    """Deterministic split of a recording by a hash of its name."""
    import hashlib
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "validation" if u < test + val else "train"
