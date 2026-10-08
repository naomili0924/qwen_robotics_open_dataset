#!/bin/bash
# Final-frame pretraining: (past frames + the frame 5 s ahead) -> the recorded positions in between.
# Inputs are images and an embodiment prompt only; the language model is fully fine-tuned, vision frozen.
#
#   bash scripts/run_ff.sh <run name> <steps> <2b|4b|2b-lora|4b-lora> [recipe] [extra vla.train flags]
#
# *-lora = LoRA (rank 32) on the language model's attention, linear-attention and MLP projections, lr 1e-4;
#          the base weights stay bf16. Later flags override earlier ones.
# recipe (optional 4th argument, default final_frame):
#   final_frame         past frames + the frame 5 s ahead -> trajectory, all source repos (ff2 / ff3 style;
#                       add --frames-repos <dedup repo> --text-loss 0.05 for ff3)
#   instruction         past frames + the sample's description / place / interaction as the task text, no final
#                       frame, the dedup repo (run ff4)
#   instruction_camera  the same plus the camera sentence from the stored calibration (--camera-prompt; run ff5)
#   indoor_instruction_camera  ff5's recipe on the private indoor sets (Aria + HM3D, 6 views, navigation instruction
#                       as the task text, stored camera sentence); 5 % of recordings / houses held out for
#                       validation; checkpoints go to a private model repo (the data's licences forbid redistribution)
# One command for a whole run with its watcher: scripts/launch_run.sh.
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
RECIPE=final_frame
case ${1:-} in final_frame|instruction|instruction_camera|indoor_instruction_camera) RECIPE=$1; shift ;; esac
DEDUP=Jinyan0924/qwen_robotics_nav_pretrain_dedup
INDOOR=Jinyan0924/qwen_robotics_nav_pretrain_dedup_indoor,Jinyan0924/hm3d_nav_instruction_clips
case $RECIPE in
  final_frame) TASK="--frames-repos $TRAIN_REPOS --frames-mix sqrt" ;;
  instruction) TASK="--frames-repos $DEDUP --val-frames-repos $DEDUP --val-split train --no-final-image --instruction description,place,interaction --text-loss 0" ;;
  instruction_camera) TASK="--frames-repos $DEDUP --val-frames-repos $DEDUP --val-split train --no-final-image --instruction description,place,interaction --text-loss 0 --camera-prompt" ;;
  indoor_instruction_camera) TASK="--frames-repos $INDOOR --val-frames-repos $INDOOR --val-split train --holdout-frac 0.05 --no-final-image --instruction instruction --text-loss 0 --camera-prompt --hub-repo Jinyan0924/qwen_robotics_nav_policy_indoor" ;;
esac
echo "run $RUN: $STEPS steps, $SIZE, recipe $RECIPE"
case $SIZE in
  2b) MODEL="--backbone Qwen/Qwen3.5-2B-Base --backbone-mode full --optimizer adamw --batch-size 16" ;;
  4b) MODEL="--backbone Qwen/Qwen3.5-4B-Base --backbone-mode full --optimizer adamw8bit --batch-size 16" ;;
  2b-lora) MODEL="--backbone Qwen/Qwen3.5-2B-Base --backbone-mode lora --lora-rank 32 --lora-alpha 64 --optimizer adamw --batch-size 16 --lr-backbone 1e-4" ;;
  4b-lora) MODEL="--backbone Qwen/Qwen3.5-4B-Base --backbone-mode lora --lora-rank 32 --lora-alpha 64 --optimizer adamw --batch-size 16 --lr-backbone 1e-4" ;;
  *) echo "size must be 2b, 4b, 2b-lora or 4b-lora"; exit 1 ;;
esac
if [ -n "${DRY_RUN:-}" ]; then echo "flags: $MODEL $TASK $*"; exit 0; fi
for attempt in 1 2 3 4 5; do
  /venv/main/bin/python -m vla.train --run /dev/shm/runs/$RUN --resume auto \
    --hub-repo Jinyan0924/qwen_robotics_nav_policy \
    --backbone-mode full --no-tune-vision --gradient-checkpointing \
    --grad-accum 1 --lr-backbone 2e-5 --lr-head 1e-4 --warmup 100 --steps $STEPS \
    $MODEL \
    --data-format frames --window-mode final_frame --stream $TASK \
    --frames 7 --past-dt-s 1.0 --horizon 10 --horizon-s 5.0 --max-pixels 112896 \
    --no-kinematic-input --no-ego-history \
    --eval-every 500 --eval-batches 8 --val-items 128 --eval-scenarios 0 \
    --save-every 500 --keep-local 1 --log-every 25 --workers 8 "$@" && break
  echo "restart $attempt after failure"; sleep 30
done
