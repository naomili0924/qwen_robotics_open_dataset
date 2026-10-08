#!/bin/bash
# The final-frame evaluation suites and their descriptions from the Hub, where the watcher and vla.eval_final_frame
# expect them: /dev/shm/final_frame_eval/{v2,v3}_final_frame and {v2,v3}_annotations.
set -eu
cd "$(dirname "$0")/.."
. scripts/train_env.sh
/venv/main/bin/python - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path("/dev/shm/final_frame_eval")
for cfg in ["v2_final_frame", "v3_final_frame", "v2_annotations", "v3_annotations"]:
    d = snapshot_download("Jinyan0924/qwen_robotics_nav_eval", repo_type="dataset", allow_patterns=[f"data/{cfg}/*"], local_dir=root / "hub")
    dest = root / cfg
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(Path(d) / "data" / cfg, dest)
    print(cfg, len(list(dest.glob("*.parquet"))), "files")
PY
