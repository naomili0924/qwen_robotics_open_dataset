"""Scenario <-> Parquet (Hugging Face `datasets`) serialisation."""
import cv2
import numpy as np
from datasets import Features, Image, List, Value

f32, seq_f32 = Value("float32"), List(Value("float32"))
steps_f32 = List(List(Value("float32")))

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
    "ego": {"x": seq_f32, "y": seq_f32, "z": seq_f32, "heading": seq_f32, "vx": seq_f32, "vy": seq_f32,
            "length": f32, "width": f32, "height": f32},
    "goal": seq_f32,
    "tracks": {
        "id": List(Value("string")), "category": List(Value("string")), "object_type": List(Value("string")),
        "length": seq_f32, "width": seq_f32, "height": seq_f32,
        "is_stationary": List(Value("bool")), "is_operator": List(Value("bool")),
        "x": steps_f32, "y": steps_f32, "z": steps_f32, "heading": steps_f32, "vx": steps_f32, "vy": steps_f32,
        "valid": List(List(Value("bool"))), "occlusion": List(List(Value("int8"))),
    },
    "tracks_to_predict": List(Value("int32")),
    "static_map": Image(),
    "observed_map": Image(),
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


def to_row(sc, static_map, observed_map, map_resolution):
    """Turn a scenario dict (numpy) into a plain-python row matching FEATURES.

    static_map / observed_map may be None for sources without lidar maps.
    """
    ego, tr = sc["ego"], sc["tracks"]
    listed = lambda a: np.asarray(a).tolist()  # noqa: E731
    at_now = tr["valid"][:, sc["current_time_index"]] if len(tr["id"]) else np.zeros(0, bool)
    future = np.stack([ego["x"], ego["y"]], 1)[sc["current_time_index"]:]
    return {
        "scenario_id": sc["scenario_id"], "dataset": sc["dataset"], "sequence": sc["sequence"],
        "segment": sc["segment"], "rate_hz": sc["rate_hz"], "current_time_index": sc["current_time_index"],
        "timestamps": listed(sc["timestamps"]), "source_frames": listed(sc["source_frames"]),
        "world_from_scenario": listed(sc["world_from_scenario"].reshape(-1)),
        "ego": {k: (listed(v) if np.ndim(v) else float(v)) for k, v in ego.items()},
        "goal": listed(sc["goal"]),
        "tracks": {k: listed(v) for k, v in tr.items()},
        "tracks_to_predict": listed(sc["tracks_to_predict"]),
        "static_map": None if static_map is None else {"bytes": png_bytes(static_map), "path": None},
        "observed_map": None if observed_map is None else {"bytes": png_bytes(observed_map), "path": None},
        "map_resolution": map_resolution,
        "num_tracks": len(tr["id"]),
        "num_pedestrians": int((at_now & (tr["object_type"] == "PEDESTRIAN") & ~tr["is_operator"]).sum()),
        "ego_speed": float(np.hypot(ego["vx"][sc["current_time_index"]], ego["vy"][sc["current_time_index"]])),
        "ego_future_distance": float(np.linalg.norm(np.diff(future, axis=0), axis=1).sum()),
    }
