# qwen_robotics_open_dataset

Converts robot social-navigation datasets into image-in, trajectory-out navigation scenarios with
3D ground truth, and scores predicted trajectories for collisions. The intended use is evaluating
humanoid navigation policies open-loop.

Each scenario is **10 past frames + current + 10 future frames**:

- **input:** front-camera images of the past and current frames, plus the robot's own past positions;
- **ground truth:** the robot's recorded future trajectory, and for the current and every future frame
  the 3D boxes of labelled objects (static or dynamic) and a lidar point cloud tagged
  ground / static / dynamic, in the style of the Waymo Open Dataset.

Data (one Hugging Face dataset per source; each card documents columns, frames and limitations):

- CODa: <https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset>
- RoboSense: <https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_robosense>
- JRDB: <https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset_jrdb>

![Example scenarios](figures/coda/coda_2hz_examples.png)

## Status

| Source | Status | Notes |
|---|---|---|
| CODa (UT Campus Object Dataset) | converted | 10 Hz labels. 1,955 scenarios at 2 Hz (5 s + 5 s), 2,603 at 10 Hz (1 s + 1 s) |
| RoboSense | converted | 1 Hz labels, so one config at 1 Hz (10 s + 10 s). Cars, pedestrians and cyclists only |
| SiT | blocked | download link is issued only after signing the authors' terms-of-use form; licence statements conflict on whether converted data may be shared |
| JRDB | converted | 15 Hz labels; 428 scenarios at 2.5 Hz (4 s + 4 s), 389 at 5 Hz, moving-robot sequences only. Pedestrians only; odometry refined by lidar scan matching |
| SCAND | not convertible as ground truth | no object boxes or tracks, so there are no future obstacle positions to score against |
| MuSoHu | not convertible as ground truth | no object boxes or tracks; recorded from a helmet on a walking person |

The reasons are spelled out in the CODa dataset card ([`dataset_card.md`](dataset_card.md)).

## Install

```bash
pip install -r requirements.txt
```

## Evaluate a planner

A prediction is the ego (x, y) at each of the 10 future steps, in the scenario frame
(origin on the ground under the robot at the current step, +x forward, +y left).

```bash
# built-in baselines: expert, constant_velocity, straight_to_goal, stationary
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz --split test \
    --baseline constant_velocity

# your own predictions: {"<scenario_id>": [[x, y], ... 10 entries], ...}
python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz --split test \
    --pred my_predictions.json --radius 0.3 --out results.json
```

From Python, streaming so nothing large is downloaded up front:

```python
from datasets import load_dataset
from hnod import eval as ev

ds = load_dataset("Jinyan0924/qwen_robotics_open_dataset", "coda_2hz", split="test", streaming=True)
for row in ds:
    images = row["past_images"]                    # 11 PIL images, oldest first, last = current
    pred = ev.baseline_constant_velocity(row)      # replace with your model: images -> (10, 2)
    print(row["scenario_id"], ev.evaluate_scenario(row, pred)["collided"])
```

The evaluator checks the path against moving boxes, stationary objects and the static obstacle map
(`collided`), and separately against the lidar points of each future step (`collided_lidar`).
Test-split baselines for `coda_2hz` (`collided`, %): recorded path 0.9, constant velocity 4.3,
straight to goal 3.7, standing still 8.4. The recorded path's 0.9% is the label-noise floor.
At 10 Hz the horizon is only 1 s and the baselines are indistinguishable, so use `coda_2hz`.

## Rebuild the CODa conversion

No account or token is needed. The full archive is 163 GB, but the pipeline reads only the parts it
needs over HTTP range requests and never stores raw images or point clouds. Intermediates take about
12 GB and the output 31 GB.

```bash
python scripts/download_coda_annotations.py --out data/raw/coda       # boxes, poses, calibration (1.3 GB)
python scripts/stream_coda_lidar.py --raw data/raw/coda --out data/interim/coda_lidar     # 40 GB streamed, 2 GB kept
python scripts/stream_coda_images.py --raw data/raw/coda --out data/interim/coda_images   # 59 GB streamed, 9 GB kept
python scripts/convert_coda.py --raw data/raw/coda --bev data/interim/coda_lidar \
    --images data/interim/coda_images --out data/hf
python scripts/visualize.py --data data/hf/coda_2hz --split test --num 3 --panels --out figures/sample.png
python -m pytest tests
```

## Rebuild the JRDB conversion

The JRDB train archive (75 GB, single zip) is read over HTTP range requests; about 14 GB of it is
needed (labels, timestamps, calibration, the forward camera, both lidars). The robot's poses are not in
the archive: the wheel odometry published with Google's Human Scene Transformer is the starting point
and is refined by lidar scan matching (needs `open3d`, `pypcd4`).

```bash
python scripts/download_jrdb.py --out data/raw/jrdb --groups labels/labels_3d,timestamps,calibration,images/image_0,pointclouds/upper_velodyne,pointclouds/lower_velodyne
curl -O https://storage.googleapis.com/gresearch/human_scene_transformer/odometry.zip && unzip -q odometry.zip -d data/raw/jrdb_odometry_src
python scripts/refine_jrdb_odometry.py --raw data/raw/jrdb --odometry data/raw/jrdb_odometry_src/odometry --out data/raw/jrdb_odometry_icp
python scripts/reduce_jrdb.py --raw data/raw/jrdb --odometry data/raw/jrdb_odometry_icp --out data/interim/jrdb_lidar --images-out data/interim/jrdb_images
python scripts/convert_jrdb.py --raw data/raw/jrdb --odometry data/raw/jrdb_odometry_icp --bev data/interim/jrdb_lidar --images data/interim/jrdb_images --out data/hf_jrdb
```

## Rebuild the RoboSense conversion

RoboSense is public on Hugging Face. Its labels are two pickle files (about 1 GB); lidar and images
are tar.gz archives of 239 GB and 163 GB in parts that can only be read front to back. The stream
scripts buffer parts in `/dev/shm` (about 35 GB of RAM disk each) and need `pigz`.

```bash
huggingface-cli download --repo-type dataset suhaisheng0527/RoboSense --include "splits/robosense_global_*.pkl" \
    --local-dir data/raw/robosense
python scripts/stream_robosense_lidar.py --pkl data/raw/robosense/splits --out data/interim/robosense_lidar
python scripts/stream_robosense_images.py --pkl data/raw/robosense/splits --out data/interim/robosense_images
python scripts/convert_robosense.py --pkl data/raw/robosense/splits --bev data/interim/robosense_lidar \
    --images data/interim/robosense_images --out data/hf_robosense
```

## Train a policy (Qwen-VL backbone + pluggable heads)

`vla/` is a small pipeline-style library for image-in, trajectory-out policies on these scenarios: a
Qwen-VL backbone reads the 11 past frames plus a short prompt (speed, past positions, goal), and heads
on its embedding emit the 10 future positions and, optionally, other quantities. Three independent
design axes are flags:

| Flag | Choices | What it decides |
|---|---|---|
| `--head` | `regression`, `flow` | regress one trajectory (optionally `--modes K` hypotheses, winner-takes-all), or sample from a flow-matching head |
| `--denoiser` | `mlp`, `dit` | flow head only: an MLP on pooled features, or a DiT with cross-attention to every backbone token |
| `--backbone` / `--backbone-mode` | any Qwen-VL checkpoint; `frozen`, `lora`, `full` | which model, and how much of it trains (`--tune-vision` includes the vision tower) |
| `--tasks` | `trajectory` plus any of `occupancy`, `collision`, `pedestrians`, `progress` | extra heads on the same embedding (see `vla/tasks.py`; adding a task is one class) |

```python
from vla.pipeline import NavigationPipeline

pipe = NavigationPipeline(data="Jinyan0924/qwen_robotics_open_dataset", config_name="coda_2hz",
                          head="flow", denoiser="dit", backbone_mode="lora", tasks="trajectory,occupancy")
pipe.fit(steps=2000, run="runs/flow_dit")                # validation metrics every 250 steps
pipe.save_pretrained("runs/flow_dit/final")

out = pipe.predict(images, ego_history, goal, rate_hz=2.0)   # 11 PIL images, (11, 2) past xy, (2,) goal
out["trajectory"]                                        # (10, 2) metres in the robot frame
out["occupancy"]                                         # (8, 8) occupancy ahead, if that head is attached

# Is the learnt embedding transferable?  Attach a new head, freeze everything else, train only the head.
pipe = NavigationPipeline.from_pretrained("runs/flow_dit/final")
pipe.add_task("collision")
pipe.freeze_backbone()
pipe.fit(steps=300, run="runs/transfer_collision")
pipe.evaluate(split="validation", scenarios=200)        # collision_auroc / collision_acc, plus the rest
```

The same from the command line:

```bash
pip install -r requirements-train.txt
python -m vla.train   --data /path/to/coda_2hz --head flow --denoiser dit --tasks trajectory,occupancy --run runs/flow_dit
python -m vla.train   --init-from runs/flow_dit/last --freeze-backbone --tasks trajectory,occupancy,collision --run runs/transfer
python -m vla.predict --checkpoint runs/flow_dit/last --split validation --out runs/flow_dit/val_pred.json

# sanity checks, a few minutes each
python -m vla.debug data     --data /path/to/coda_2hz --run runs/debug           # prompts, token counts, a figure
python -m vla.debug forward  --data /path/to/coda_2hz --run runs/debug           # losses, grad norms, GPU memory
python -m vla.debug compare  --data /path/to/coda_2hz --run runs/debug --steps 150   # overfit 8 scenarios with every head
python -m vla.debug pipeline --data /path/to/coda_2hz --run runs/debug --steps 20    # fit / save / reload / add head / predict
```

Checkpoints hold the heads, the LoRA adapter (or full backbone weights) and the config. Logs go to
`<run>/log.jsonl` and TensorBoard; validation metrics are the collision evaluator's (`collided`,
`collided_lidar`, ADE, FDE) plus each task's own. On one H100, Qwen2.5-VL-3B with LoRA at 448 px per
frame takes about 13 GB at batch 4 and 2 s per step.

### PPO / GRPO fine-tuning

`vla.rl` fine-tunes a policy with reinforcement learning, TRL-style, using the collision evaluator as the
reward. One scenario is one decision (a whole trajectory), i.e. the single-step setting PPO/GRPO are used
in for language models:

- **Policy distribution.** Regression head: a Gaussian around the predicted trajectory (a mixture if
  `--modes K`) with a learnable std. Flow head: the ODE sampler is replaced by an SDE with the same
  marginals (as in Flow-GRPO), so every denoising step is a Gaussian transition with an exact log-prob.
- **Rollout.** `--group-size` trajectories per scenario, each scored by `vla/rewards.py`:
  `collision` (+1 / -1 from the evaluator, lidar check included), `goal`, `imitation`, `smooth`,
  `progress`; combine with `--reward "collision=1,goal=0.5,imitation=0.2"`. Adding a term is one function.
- **Update.** GRPO normalises rewards within each scenario's group; PPO uses a value head on the embedding.
  Both use the clipped ratio, a KL penalty against the reference policy (the weights RL started from: a
  frozen copy of the head and, for LoRA, a frozen snapshot adapter) and an optional supervised term
  (`--bc-weight`). Dropout is switched off during RL so rollout and update log-probs agree.

```bash
python -m vla.rl --algo grpo --init-from runs/flow_dit/last --head flow --denoiser dit \
    --group-size 8 --reward "collision=1,goal=0.5,imitation=0.2,smooth=0.1" --run runs/grpo --steps 300
python -m vla.rl --algo ppo  --init-from runs/sft/last --head regression --modes 3 --bc-weight 0.5 --run runs/ppo
```

or `pipe.fit_rl(algo="grpo", steps=300, run="runs/grpo")`. RL uses its own learning rates
(`--rl-lr-head`, `--rl-lr-backbone`, default 1e-5): the SFT rates make a narrow Gaussian policy jump
to KL values in the hundreds within a few steps. Logged per step: mean reward and each term, in-group
reward std, KL, clip fraction, value loss; the usual validation metrics at `--eval-every`.

### Is the embedding good enough for an MLP head?

`vla.probe` answers that before any training. It freezes the backbone, extracts pooled features at
every layer (last prompt token, mean of image tokens, mean of text tokens), fits a linear probe and a
small MLP probe on each, and compares them with kinematics-only probes (past positions, speed, goal)
on three targets: the future trajectory, the *residual* of the trajectory against the straight line
to the goal (the part that needs the scene), and an 8 x 8 occupancy grid of the 8 m ahead (whether
the embedding knows where obstacles are).

```bash
python -m vla.probe --data /path/to/coda_2hz --run runs/probe --train-samples 600 --val-samples 200
```

It writes `probe.png` (metric vs layer) and `probe_summary.txt`. Linear close to MLP: the feature is
linearly usable, an MLP head is enough. MLP far better than linear: the information is there but
entangled, a deeper head helps. Neither beating the kinematic baseline: the frozen embedding lacks the
information, so adapt the backbone (`--backbone-mode lora`) or give the head token-level access
(`--denoiser dit`). Best layer well before the last: tap an earlier layer.

Every Config field in `vla/config.py` is a flag. Logs go to `<run>/log.jsonl` and TensorBoard; the
validation metrics are the collision evaluator's (`collided`, `collided_lidar`, ADE, FDE). On one H100,
Qwen2.5-VL-3B with LoRA at 448 px per frame takes about 13 GB at batch 4 and 2 s per step.

## Layout

| Path | Purpose |
|---|---|
| `hnod/coda.py`, `hnod/robosense.py`, `hnod/jrdb.py` | source readers: labelled segments with world-frame ego poses, boxes and camera calibration |
| `scripts/refine_jrdb_odometry.py` | lidar scan matching to replace JRDB's drifting wheel odometry |
| `hnod/scenario.py` | track clean-up, operator detection, windowing into ego-centric scenarios |
| `hnod/lidar_bev.py` | lidar sweep to ground/obstacle grid and thinned, ground-flagged points |
| `hnod/points.py` | per-step point clouds in the scenario frame, labelled ground / static / dynamic |
| `hnod/maps.py` | merge grids into the per-scenario static map |
| `hnod/pipeline.py` | segment to Parquet rows, shared by all converters |
| `hnod/io.py` | Parquet schema |
| `hnod/eval.py` | collision evaluator and baselines |
| `scripts/` | download, stream, convert, evaluate, visualise, publish |
| `vla/` | policy training: `data.py` prompts and targets, `heads.py` regression / flow heads with MLP or DiT denoisers (+ stochastic policies), `model.py` backbone + LoRA, `tasks.py` auxiliary heads, `train.py` supervised, `rl.py` PPO / GRPO with `rewards.py`, `pipeline.py` the front end, `predict.py`, `debug.py` pipeline checks, `probe.py` representation probing |

Adding another source means writing a reader that returns segments in the layout documented in
`hnod.coda.load_segments`; everything downstream is shared.

## License

Code: MIT (see `LICENSE`). Converted data: CC BY-NC-SA 4.0 for CODa and RoboSense, CC BY-NC-SA 3.0 for
JRDB, inherited from the sources; cite the source dataset when using it (citations in the dataset cards).
