# sourced by training runs: credentials, no Hub chunk cache (bounded disk), quiet progress bars
set -a; . /workspace/.env; set +a
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TRAIN_REPOS=Jinyan0924/qwen_robotics_open_dataset_egowalk,Jinyan0924/habitat_hssd_pointgoal_nav_scenarios,Jinyan0924/qwen_robotics_open_dataset_robosense,Jinyan0924/qwen_robotics_open_dataset
