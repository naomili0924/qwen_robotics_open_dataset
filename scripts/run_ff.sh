#!/bin/bash
# Final-frame pretraining: (past frames + the frame 5 s ahead) -> the recorded positions in between.
# Inputs are images and "You are a human walking." / "You are a robot." only; the language model is fully fine-tuned.
#   bash scripts/run_ff.sh <run name> <steps> [extra vla.train flags]
# Resumes from /dev/shm/runs/<run name>/last, or from the Hub copy on a new machine.
set -u
cd "$(dirname "$0")/.."
. scripts/train_env.sh
RUN=${1:?run name}; STEPS=${2:?steps}; shift 2
for attempt in 1 2 3 4 5; do
  /venv/main/bin/python -m vla.train --run /dev/shm/runs/$RUN --resume auto \
    --hub-repo Jinyan0924/qwen_robotics_nav_policy \
    --backbone Qwen/Qwen3-VL-2B-Instruct --backbone-mode full --no-tune-vision --no-gradient-checkpointing \
    --attn-implementation kernels-community/flash-attn3 \
    --data-format frames --window-mode final_frame --stream --frames-repos $TRAIN_REPOS --frames-mix sqrt \
    --frames 7 --past-dt-s 1.0 --horizon 10 --horizon-s 5.0 --max-pixels 112896 \
    --no-kinematic-input --no-ego-history \
    --batch-size 16 --grad-accum 1 --lr-backbone 2e-5 --lr-head 1e-4 --warmup 100 --steps $STEPS \
    --eval-every 250 --eval-batches 8 --val-items 128 --eval-scenarios 0 \
    --save-every 250 --keep-local 1 --log-every 25 --workers 8 "$@" && break
  echo "restart $attempt after failure"; sleep 30
done
