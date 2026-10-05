#!/bin/bash
# First end-to-end model: all training sources streamed from the Hub, one epoch. Restarts and resumes on a crash.
cd /workspace/qwen_robotics_open_dataset
. scripts/train_env.sh
for attempt in 1 2 3 4 5; do
  /venv/main/bin/python -m vla.train --run /workspace/runs/e1_all_sqrt --resume auto \
    --hub-repo Jinyan0924/qwen_robotics_nav_policy \
    --data-format frames --stream --frames-repos $TRAIN_REPOS --frames-mix sqrt \
    --frames 6 --past-dt-s 1.0 --max-pixels 112896 --horizon 8 --spacing-m 0.25 \
    --attn-implementation kernels-community/flash-attn3 --lora-dropout 0.0 --no-gradient-checkpointing \
    --batch-size 8 --grad-accum 2 --steps 69400 --warmup 500 \
    --eval-every 2000 --eval-batches 32 --val-items 256 --eval-scenarios 0 \
    --save-every 1000 --keep-local 2 --log-every 50 --workers 6
  grep -q '^done:' /workspace/logs/e1.log && break
  echo "restart $attempt after failure" ; sleep 60
done
