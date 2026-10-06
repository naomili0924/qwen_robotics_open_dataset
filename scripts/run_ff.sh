#!/bin/bash
# Final-frame pretraining: (past frames + the frame 5 s ahead) -> the recorded positions in between.
# Inputs are images and an embodiment prompt only; the language model is fully fine-tuned, vision frozen.
#
#   bash scripts/run_ff.sh <run name> <steps> <2b|4b> [extra vla.train flags]
#
# 2b = Qwen/Qwen3.5-2B-Base: one 94 GB H100, batch 16, about 4 samples/s (gradient checkpointing, 39 GB).
# 4b = Qwen/Qwen3.5-4B-Base: the Qwen-VLA backbone; on one 94 GB GPU it needs 8-bit AdamW and runs at about
#      1.8 samples/s (44 GB). On a bigger GPU add "--optimizer adamw --no-gradient-checkpointing" for speed.
# Resumes from /dev/shm/runs/<run name>/last, or from the Hub copy on a new machine. Needs HF_TOKEN in /workspace/.env.
set -u
cd "$(dirname "$0")/.."
. scripts/train_env.sh
export HF_HUB_CACHE=${HF_HUB_CACHE:-/dev/shm/hf_hub}   # base-model weights on the RAM disk (the root disk is small)
RUN=${1:?run name}; STEPS=${2:?steps}; SIZE=${3:?2b|4b}; shift 3
case $SIZE in
  2b) MODEL="--backbone Qwen/Qwen3.5-2B-Base --optimizer adamw --batch-size 16" ;;
  4b) MODEL="--backbone Qwen/Qwen3.5-4B-Base --optimizer adamw8bit --batch-size 16" ;;
  *) echo "size must be 2b or 4b"; exit 1 ;;
esac
for attempt in 1 2 3 4 5; do
  /venv/main/bin/python -m vla.train --run /dev/shm/runs/$RUN --resume auto \
    --hub-repo Jinyan0924/qwen_robotics_nav_policy \
    $MODEL --backbone-mode full --no-tune-vision --gradient-checkpointing \
    --data-format frames --window-mode final_frame --stream --frames-repos $TRAIN_REPOS --frames-mix sqrt \
    --frames 7 --past-dt-s 1.0 --horizon 10 --horizon-s 5.0 --max-pixels 112896 \
    --no-kinematic-input --no-ego-history \
    --grad-accum 1 --lr-backbone 2e-5 --lr-head 1e-4 --warmup 100 --steps $STEPS \
    --eval-every 500 --eval-batches 8 --val-items 128 --eval-scenarios 0 \
    --save-every 500 --keep-local 1 --log-every 25 --workers 8 "$@" && break
  echo "restart $attempt after failure"; sleep 30
done
