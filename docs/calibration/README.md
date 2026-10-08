# Camera calibration and its provenance

Per-episode and per-scenario camera calibration as given to the model's prompt (`hnod.windows.camera_prompt`), kept here
so it never has to be re-derived. Rule (owner, 2026-10-06): **only stored calibration is used; nothing is estimated.**
A height whose source is not the dataset's own calibration or a simulator is `null` with `height_source = "unknown"`,
and the prompt says so.

- `episodes_camera.jsonl`: one line per training episode (EgoWalk, CODa, RoboSense, HSSD; the dedup repo reuses these):
  `repo, dataset, episode_id, name, width, height, K (3x3 row-major), distortion_model, distortion, height_m,
  height_source, camera_prompt`.
- `eval_camera.jsonl`: one line per scenario of `v2_final_frame` and `v3_final_frame`: `suite, suite_id, scenario_id,
  source, name, width, height, K, height_m, height_source, camera_prompt`.

Sources of the height above the ground: EgoWalk `meta/heights.json` of the dataset (`dataset`); CODa, JRDB,
RoboSense from their extrinsic calibration (`dataset`); HSSD rendered at 1.2 m (`simulator`); MuSoHu's helmet height
was estimated from the lidar during conversion and is therefore `unknown`. Field of view = 2·atan(width / (2·fx)).

Regenerate from the Hub tables with `python scripts/add_camera_provenance.py` (the export is the last step).
