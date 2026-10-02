"""Task registry: what a head predicts, from which row columns, with which loss and metric.

`trajectory` is the main task and is served by the action head (regression or flow).
Every other task is an auxiliary head on the same embedding; attach any of them with
`--tasks trajectory,occupancy,collision` or `pipeline.add_task("occupancy")`.  To
measure how transferable a learnt embedding is, attach a task to a trained model with
the backbone frozen and train only the new head.

Adding a task: subclass Task, implement target()/loss()/metrics(), register it in TASKS.
"""
import numpy as np
import torch
import torch.nn.functional as F

from hnod import eval as ev
from hnod.eval import decode_map
from hnod.maps import MAP_RES, MAP_SIZE, PIX_OCCUPIED
from hnod.scenario import CURRENT, N_FUTURE


class Task:
    name = ""
    columns = ()      # dataset columns needed beyond the inputs
    out_dim = 1
    weight = 1.0      # loss weight

    def target(self, row, cfg):
        raise NotImplementedError

    def loss(self, pred, target):
        raise NotImplementedError

    def metrics(self, pred, target):
        """pred, target: numpy arrays (N, out_dim)."""
        raise NotImplementedError

    def decode(self, pred):
        """Model output -> what a user wants back (numpy, per sample)."""
        return pred


class OccupancyTask(Task):
    """Fraction of occupied map cells in an 8 x 8 grid over the 8 m ahead (x in [0, 8], y in [-4, 4])."""
    name = "occupancy"
    columns = ("static_map",)
    cells, extent = 8, 8.0
    out_dim = 64

    def target(self, row, cfg):
        occ = (decode_map(row["static_map"]) == PIX_OCCUPIED).astype(np.float32)
        half = MAP_SIZE // 2
        r0, c0 = int(half - self.extent / MAP_RES), int(half - self.extent / 2 / MAP_RES)
        n = int(self.extent / MAP_RES)
        block = occ[r0:half, c0:c0 + n]
        k = n // self.cells
        return block.reshape(self.cells, k, self.cells, k).mean((1, 3)).ravel()

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def metrics(self, pred, target):
        p = 1 / (1 + np.exp(-pred))
        ss = ((target - p) ** 2).sum()
        return {"occupancy_r2": float(1 - ss / max(((target - target.mean(0)) ** 2).sum(), 1e-9)),
                "occupancy_mae": float(np.abs(target - p).mean())}

    def decode(self, pred):
        return (1 / (1 + np.exp(-pred))).reshape(self.cells, self.cells)


class CollisionTask(Task):
    """Would the straight line to the goal collide (per the collision evaluator)? Binary."""
    name = "collision"
    columns = ("future_tracks", "future_lidar", "static_map", "ego", "goal", "rate_hz", "scenario_id")
    out_dim = 1

    def target(self, row, cfg):
        res = ev.evaluate_scenario(row, ev.baseline_straight_to_goal(row))
        return np.array([float(res["collided"] or res["collided_lidar"])], np.float32)

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def metrics(self, pred, target):
        p, y = 1 / (1 + np.exp(-pred[:, 0])), target[:, 0]
        out = {"collision_acc": float(((p > 0.5) == (y > 0.5)).mean()), "collision_rate": float(y.mean())}
        if 0 < y.sum() < len(y):  # AUROC by rank statistic
            order = np.argsort(p)
            ranks = np.empty(len(p))
            ranks[order] = np.arange(1, len(p) + 1)
            n1 = y.sum()
            out["collision_auroc"] = float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))
        return out

    def decode(self, pred):
        return float(1 / (1 + np.exp(-pred[0])))


class PedestrianCountTask(Task):
    """How many pedestrians are around now (log1p regression)."""
    name = "pedestrians"
    columns = ("num_pedestrians",)
    out_dim = 1

    def target(self, row, cfg):
        return np.array([np.log1p(row["num_pedestrians"])], np.float32)

    def loss(self, pred, target):
        return F.smooth_l1_loss(pred, target)

    def metrics(self, pred, target):
        return {"pedestrians_mae": float(np.abs(np.expm1(pred) - np.expm1(target)).mean())}

    def decode(self, pred):
        return float(np.expm1(pred[0]))


class ProgressTask(Task):
    """Distance the robot will actually cover over the horizon, as a fraction of the straight-line goal distance."""
    name = "progress"
    columns = ("ego", "goal")
    out_dim = 1

    def target(self, row, cfg):
        ego = row["ego"]
        xy = np.stack([ego["x"], ego["y"]], 1)[CURRENT:CURRENT + 1 + N_FUTURE]
        travelled = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        return np.array([travelled / max(np.linalg.norm(row["goal"]), 0.5)], np.float32)

    def loss(self, pred, target):
        return F.smooth_l1_loss(pred, target)

    def metrics(self, pred, target):
        return {"progress_mae": float(np.abs(pred - target).mean())}

    def decode(self, pred):
        return float(pred[0])


TASKS = {t.name: t for t in (OccupancyTask(), CollisionTask(), PedestrianCountTask(), ProgressTask())}


def aux_tasks(names):
    """Task objects for the auxiliary names in a comma-separated task list (trajectory excluded)."""
    out = []
    for n in [s.strip() for s in names.split(",") if s.strip()]:
        if n == "trajectory":
            continue
        if n not in TASKS:
            raise KeyError(f"unknown task {n!r}; known: trajectory, {', '.join(TASKS)}")
        out.append(TASKS[n])
    return out
