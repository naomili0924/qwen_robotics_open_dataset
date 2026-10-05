"""Scenario <-> Parquet (Hugging Face `datasets`) serialisation."""
import cv2
import numpy as np
from datasets import Features, Image, List, Value

f32, seq_f32 = Value("float32"), List(Value("float32"))
steps_f32 = List(List(Value("float32")))

_track_fields = {
    "id": List(Value("string")), "category": List(Value("string")), "object_type": List(Value("string")),
    "length": seq_f32, "width": seq_f32, "height": seq_f32,
    "is_stationary": List(Value("bool")), "is_operator": List(Value("bool")),
    "x": steps_f32, "y": steps_f32, "z": steps_f32, "heading": steps_f32, "vx": steps_f32, "vy": steps_f32,
    "valid": List(List(Value("bool"))), "occlusion": List(List(Value("int8"))),
}

FEATURES = Features({
    "scenario_id": Value("string"),
    "dataset": Value("string"),
    "sequence": Value("string"),
    "segment": Value("int32"),
    "rate_hz": f32,
    "current_time_index": Value("int32"),
    "timestamps": List(Value("float64")),
    "source_frames": List(Value("int32")),
    "world_from_scenario": List(Value("float64")),
    # --- model input: the past N_PAST steps and the current one
    "past_images": List(Image()),
    "camera": {"name": Value("string"), "width": Value("int32"), "height": Value("int32"),
               "K": List(Value("float64")), "T_scenario_from_camera": List(List(Value("float64")))},
    "ego": {"x": seq_f32, "y": seq_f32, "z": seq_f32, "heading": seq_f32, "vx": seq_f32, "vy": seq_f32,
            "length": f32, "width": f32, "height": f32},
    "goal": seq_f32,
    # what the robot is asked to do: "pointgoal" (reach `goal`), "vln" (follow `instruction`),
    # "objectnav" (find the object named in `instruction`); `goal` is always the end of the recorded future
    "task": Value("string"),
    "instruction": Value("string"),
    # --- ground truth: the current step and the N_FUTURE future ones
    "future_tracks": _track_fields,
    "tracks_to_predict": List(Value("int32")),
    "future_lidar": {"x": List(List(Value("int16"))), "y": List(List(Value("int16"))), "z": List(List(Value("int16"))),
                     "label": List(List(Value("uint8"))), "track": List(List(Value("int16")))},
    "static_map": Image(),
    "map_resolution": f32,
    "num_tracks": Value("int32"),
    "num_pedestrians": Value("int32"),
    "ego_speed": f32,
    "ego_future_distance": f32,
})


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def to_row(sc, static_map, map_resolution, images=None, camera=None, lidar=None):
    """Turn a scenario dict (numpy) into a plain-python row matching FEATURES.

    Track arrays are cut to the current and future steps (index 0 = current step).
    images: encoded image bytes for steps 0..current, oldest first.  camera: dict
    matching FEATURES["camera"].  lidar: output of points.scenario_points.  Any of
    static_map / images / camera / lidar may be None for sources that lack them.
    """
    ego, tr, now = sc["ego"], sc["tracks"], sc["current_time_index"]
    listed = lambda a: np.asarray(a).tolist()  # noqa: E731
    future = {k: listed(v[:, now:] if np.ndim(v) == 2 else v) for k, v in tr.items()}
    at_now = tr["valid"][:, now] if len(tr["id"]) else np.zeros(0, bool)
    path = np.stack([ego["x"], ego["y"]], 1)[now:]
    return {
        "scenario_id": sc["scenario_id"], "dataset": sc["dataset"], "sequence": sc["sequence"],
        "segment": sc["segment"], "rate_hz": sc["rate_hz"], "current_time_index": now,
        "timestamps": listed(sc["timestamps"]), "source_frames": listed(sc["source_frames"]),
        "world_from_scenario": listed(sc["world_from_scenario"].reshape(-1)),
        "past_images": None if images is None else [{"bytes": b, "path": None} for b in images],
        "camera": camera,
        "ego": {k: (listed(v) if np.ndim(v) else float(v)) for k, v in ego.items()},
        "goal": listed(sc["goal"]),
        "task": sc.get("task", "pointgoal"), "instruction": sc.get("instruction", ""),
        "future_tracks": future,
        "tracks_to_predict": listed(sc["tracks_to_predict"]),
        "future_lidar": None if lidar is None else {k: [a.tolist() for a in v] for k, v in lidar.items()},
        "static_map": None if static_map is None else {"bytes": png_bytes(static_map), "path": None},
        "map_resolution": map_resolution,
        "num_tracks": len(tr["id"]),
        "num_pedestrians": int((at_now & (tr["object_type"] == "PEDESTRIAN") & ~tr["is_operator"]).sum()),
        "ego_speed": float(np.hypot(ego["vx"][now], ego["vy"][now])),
        "ego_future_distance": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
    }
