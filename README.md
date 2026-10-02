# qwen_robotics_open_dataset

Converts robot social-navigation datasets into Waymo-Open-Motion-style scenarios
(**10 past steps + current + 10 future steps**) and scores predicted ego trajectories for
collisions with moving agents, stationary objects and static structure. The intended use
is evaluating humanoid navigation policies open-loop.

- **Data** (one Hugging Face dataset per source; schema, coordinate conventions, evaluation protocol
  and limitations are in each dataset card):
  - CODa: <https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset>
  - RoboSense: <https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_robosense>
- **This repository:** the conversion pipeline and the evaluator

![Example scenarios](figures/coda_2hz_examples.png)

## Status

| Source | Status | Notes |
|---|---|---|
| CODa (UT Campus Object Dataset) | converted | 10 Hz labels. 1,961 scenarios at 2 Hz (5 s + 5 s), 5,244 at 10 Hz (1 s + 1 s) |
| RoboSense | converted | 1 Hz labels, so one config at 1 Hz (10 s + 10 s). Cars, pedestrians and cyclists only |
| SiT | blocked | download link is issued only after signing the authors' terms-of-use form; licence statements conflict on whether converted data may be shared |
| JRDB | draft converter, untested | needs a registered JRDB account to download labels; `scripts/convert_jrdb.py` has only been run on synthetic labels |
| SCAND | not convertible as ground truth | no object boxes or tracks, so there are no future obstacle positions to score against |
| MuSoHu | not convertible as ground truth | no object boxes or tracks; recorded from a helmet on a walking person |

The reasons are spelled out in the CODa dataset card ([`dataset_card.md`](dataset_card.md)).

## Install

```bash
pip install -r requirements.txt
```

## Evaluate a planner

A prediction is the ego (x, y) at each of the 10 future steps, in the scenario frame
(origin at the ego at the current step, +x forward, +y left).

```bash
# built-in baselines: expert, constant_velocity, straight_to_goal, stationary
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz --split test \
    --baseline constant_velocity

# your own predictions: {"<scenario_id>": [[x, y], ... 10 entries], ...}
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz --split test \
    --pred my_predictions.json --radius 0.3 --out results.json
```

From Python:

```python
from datasets import load_dataset
from hnod import eval as ev

ds = load_dataset("Jinyan0924/qwen_robotics_open_dataset", "coda_2hz", split="test")
results = [ev.evaluate_scenario(row, ev.baseline_constant_velocity(row)) for row in ds]
print(ev.aggregate(results))
```

Test-split baselines for `coda_2hz` (collision rate, %): recorded path 0.9, constant velocity 4.3,
straight to goal 3.7, standing still 8.4. The recorded path's 0.9% is the label-noise floor.
At 10 Hz the horizon is only 1 s and all baselines are within 0.3-1.6%, so use `coda_2hz` for evaluation.

## Rebuild the CODa conversion

No account or token is needed for CODa. The full archive is 163 GB, but the pipeline reads only
what it needs over HTTP range requests and never stores images or point clouds; about 3 GB of
disk is enough.

```bash
python scripts/download_coda_annotations.py --out data/raw/coda        # boxes, poses, timestamps, terrain labels (1.3 GB)
python scripts/stream_coda_lidar.py --raw data/raw/coda --out data/interim/coda_bev   # 40 GB streamed, 0.5 GB kept
python scripts/convert_coda.py --raw data/raw/coda --bev data/interim/coda_bev --out data/hf
python scripts/visualize.py --data data/hf/coda_2hz --split test --num 6 --out figures/sample.png
python -m pytest tests
```

## Rebuild the RoboSense conversion

RoboSense is public on Hugging Face. Its labels are two pickle files (about 1 GB); the lidar is one
239 GB tar.gz in 23 parts that can only be read front to back. The stream script buffers parts in
`/dev/shm` (about 45 GB of RAM disk), needs `pigz`, and keeps only 0.5 GB of grids.

```bash
huggingface-cli download --repo-type dataset suhaisheng0527/RoboSense --include "splits/robosense_global_*.pkl" \
    --local-dir data/raw/robosense
python scripts/stream_robosense_lidar.py --pkl data/raw/robosense/splits --out data/interim/robosense_bev
python scripts/convert_robosense.py --pkl data/raw/robosense/splits --bev data/interim/robosense_bev --out data/hf_robosense
```

## Layout

| Path | Purpose |
|---|---|
| `hnod/coda.py`, `hnod/robosense.py`, `hnod/jrdb.py` | source readers: labelled segments with world-frame ego poses and boxes |
| `hnod/scenario.py` | track clean-up, operator detection, windowing into ego-centric scenarios |
| `hnod/lidar_bev.py` | lidar sweep to ground/obstacle grid |
| `hnod/maps.py` | merge grids into the per-scenario static map |
| `hnod/pipeline.py` | segment to Parquet rows, shared by all converters |
| `hnod/io.py` | Parquet schema |
| `hnod/eval.py` | collision evaluator and baselines |
| `scripts/` | download, convert, evaluate, visualise, publish |

Adding another source means writing a reader that returns segments in the layout documented in
`hnod.coda.load_segments`; everything downstream is shared.

## License

Code: MIT (see `LICENSE`). Converted data: CC BY-NC-SA 4.0, inherited from CODa and RoboSense; cite the
source dataset when using it (citations in the dataset cards).
