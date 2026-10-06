#!/bin/bash
# Final-frame pretraining: (past frames + the frame 5 s ahead) -> the recorded positions in between.
# Inputs are images and an embodiment prompt only; the language model is fully fine-tuned, vision frozen.
#
#   bash scripts/run_ff.sh <run name> <steps> <2b|4b|2b-lora|4b-lora> [extra vla.train flags]
#
# *-lora = LoRA (rank 32) on the language model's attention, linear-attention and MLP projections, lr 1e-4;
#          the base weights stay bf16. Later flags override earlier ones (e.g. --frames-repos <dedup repo>,
#          --text-loss 0.05 to also predict each sample's description from data/annotations).
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
  2b) MODEL="--backbone Qwen/Qwen3.5-2B-Base --backbone-mode full --optimizer adamw --batch-size 16" ;;
  4b) MODEL="--backbone Qwen/Qwen3.5-4B-Base --backbone-mode full --optimizer adamw8bit --batch-size 16" ;;
  2b-lora) MODEL="--backbone Qwen/Qwen3.5-2B-Base --backbone-mode lora --lora-rank 32 --lora-alpha 64 --optimizer adamw --batch-size 16 --lr-backbone 1e-4" ;;
  4b-lora) MODEL="--backbone Qwen/Qwen3.5-4B-Base --backbone-mode lora --lora-rank 32 --lora-alpha 64 --optimizer adamw --batch-size 16 --lr-backbone 1e-4" ;;
  *) echo "size must be 2b, 4b, 2b-lora or 4b-lora"; exit 1 ;;
esac
for attempt in 1 2 3 4 5; do
  /venv/main/bin/python -m vla.train --run /dev/shm/runs/$RUN --resume auto \
    --hub-repo Jinyan0924/qwen_robotics_nav_policy \
    --backbone-mode full --no-tune-vision --gradient-checkpointing \
    --grad-accum 1 --lr-backbone 2e-5 --lr-head 1e-4 --warmup 100 --steps $STEPS \
    $MODEL \
    --data-format frames --window-mode final_frame --stream --frames-repos $TRAIN_REPOS --frames-mix sqrt \
    --frames 7 --past-dt-s 1.0 --horizon 10 --horizon-s 5.0 --max-pixels 112896 \
    --no-kinematic-input --no-ego-history \
    --eval-every 500 --eval-batches 8 --val-items 128 --eval-scenarios 0 \
    --save-every 500 --keep-local 1 --log-every 25 --workers 8 "$@" && break
  echo "restart $attempt after failure"; sleep 30
done
