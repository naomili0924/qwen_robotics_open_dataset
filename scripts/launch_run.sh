#!/bin/bash
# One command for a complete run: training (resumable, checkpoints every 500 steps to the Hub) plus the watcher
# that uploads a milestone every 1,000 steps and scores it on the v2 and v3 final-frame suites.
#
#   bash scripts/launch_run.sh <run name> <steps> <2b|4b|2b-lora|4b-lora> <final_frame|instruction|instruction_camera> [extra flags]
#
#   bash scripts/launch_run.sh ff4_lora4b_instruction 6200 4b-lora instruction          # past frames + text
#   bash scripts/launch_run.sh ff5_lora4b_instruction_camera 6200 4b-lora instruction_camera   # + camera sentence
#
# Logs: /workspace/logs/<run>.log and /workspace/logs/watch_<run>.log.  Re-running the same command resumes the run
# from <run>/last (locally or from the Hub).  Needs HF_TOKEN in /workspace/.env and the eval suites under
# /dev/shm/final_frame_eval/{v2,v3}_final_frame and {v2,v3}_annotations (scripts/fetch_eval_suites.sh).
set -u
cd "$(dirname "$0")/.."
RUN=${1:?run name}; STEPS=${2:?steps}; SIZE=${3:?size}; RECIPE=${4:?recipe}; shift 4
mkdir -p /workspace/logs
[ -d /dev/shm/final_frame_eval/v3_final_frame ] || bash scripts/fetch_eval_suites.sh
nohup bash scripts/run_ff.sh "$RUN" "$STEPS" "$SIZE" "$RECIPE" --save-every 500 --eval-every 500 "$@" >> "/workspace/logs/$RUN.log" 2>&1 &
echo "training started (pid $!), log /workspace/logs/$RUN.log"
sleep 120
. scripts/train_env.sh
export HF_HUB_CACHE=${HF_HUB_CACHE:-/dev/shm/hf_hub}
MILESTONES=$(seq -s, 1000 1000 "$STEPS")
nohup /venv/main/bin/python scripts/watch_checkpoints.py --run "/dev/shm/runs/$RUN" --hub-repo Jinyan0924/qwen_robotics_nav_policy \
  --milestones "$MILESTONES" --every 100000 --suite v2_final_frame v3_final_frame \
  --suite-data "/dev/shm/final_frame_eval/{version}" >> "/workspace/logs/watch_$RUN.log" 2>&1 &
echo "watcher started (pid $!), milestones $MILESTONES, log /workspace/logs/watch_$RUN.log"
