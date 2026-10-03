"""Reward terms for RL fine-tuning; combine them with a spec like "collision=1,goal=0.5,imitation=0.2".

Each term maps (row, trajectory in metres) -> float, higher is better, roughly in [-1, 1].
Adding one: write a function, register it in REWARDS.
"""
import numpy as np

from hnod import eval as ev
from hnod.scenario import CURRENT


def _scale(row):
    return max(1.0, float(np.linalg.norm(row["goal"])))


def collision(row, traj, res):
    """+1 if the path hits nothing (boxes, map and per-step lidar), else -1."""
    return -1.0 if (res["collided"] or res["collided_lidar"]) else 1.0


def goal(row, traj, res):
    """Negative final distance to the goal, relative to the goal distance."""
    return -float(np.linalg.norm(traj[-1] - np.asarray(row["goal"]))) / _scale(row)


def imitation(row, traj, res):
    """Negative ADE against the recorded path, relative to the goal distance."""
    return -res["ade"] / _scale(row)


def smooth(row, traj, res):
    """Negative mean acceleration magnitude (second differences), per metre of goal distance."""
    p = np.vstack([[row["ego"]["x"][CURRENT], row["ego"]["y"][CURRENT]], traj])
    return -float(np.linalg.norm(np.diff(p, 2, axis=0), axis=1).mean()) / _scale(row)


def progress(row, traj, res):
    """How much closer to the goal the path ends than it started, relative to the goal distance (clipped)."""
    d0 = float(np.linalg.norm(row["goal"]))
    return float(np.clip((d0 - np.linalg.norm(traj[-1] - np.asarray(row["goal"]))) / _scale(row), -1, 1))


REWARDS = {f.__name__: f for f in (collision, goal, imitation, smooth, progress)}


def parse_spec(spec):
    weights = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        name, _, w = item.partition("=")
        if name.strip() not in REWARDS:
            raise KeyError(f"unknown reward term {name!r}; known: {', '.join(REWARDS)}")
        weights[name.strip()] = float(w) if w else 1.0
    return weights


def compute(row, traj, weights, radius=ev.DEFAULT_RADIUS):
    """Weighted reward and its components for one trajectory (horizon, 2) in metres."""
    res = ev.evaluate_scenario(row, traj, radius=radius)
    parts = {name: REWARDS[name](row, traj, res) for name in weights}
    return sum(weights[k] * v for k, v in parts.items()), parts
