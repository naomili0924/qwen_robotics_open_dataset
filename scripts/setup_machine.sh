#!/bin/bash
# Bring a fresh machine to the point where data generation and training can resume.
# Nothing here depends on the previous machine: code comes from GitHub, datasets and
# checkpoints from the Hugging Face Hub, credentials from ${WORKSPACE}/.env (which you supply).
#
#   git clone https://github.com/naomili0924/qwen_robotics_open_dataset && cd qwen_robotics_open_dataset
#   bash scripts/setup_machine.sh            # add --habitat to also install the simulator environment
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE="${WORKSPACE:-/workspace}/.env"

echo "== credentials (${ENV_FILE})"
for key in HF_TOKEN GITHUB_TOKEN; do
    grep -q "^${key}=" "$ENV_FILE" 2>/dev/null && echo "  $key present" || echo "  $key MISSING: add ${key}=... to ${ENV_FILE}"
done
for key in MATTERPORT_TOKEN_ID MATTERPORT_TOKEN_SECRET; do
    grep -q "^${key}=" "$ENV_FILE" 2>/dev/null && echo "  $key present" || echo "  $key not set (only needed for HM3D)"
done

echo "== storage"
if command -v vast-capabilities >/dev/null; then
    vast-capabilities | python3 -c "import json,sys; d=json.load(sys.stdin); print('  workspace is a persistent volume:', d['instance'].get('workspace_is_volume'))"
fi
df -h / /dev/shm "${WORKSPACE:-/workspace}" 2>/dev/null | sed 's/^/  /'

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | sed 's/^/  /' || true

echo "== python environment (conversion, evaluation, training)"
PY=python3
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
$PY -m pip install -q -r requirements-train.txt
$PY -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ "${1:-}" = "--habitat" ]; then
    echo "== habitat-sim environment (/venv/habitat)"
    export PATH=/opt/miniforge3/bin:$PATH
    [ -x /venv/habitat/bin/python ] || conda create -y -q -p /venv/habitat python=3.9 habitat-sim=0.3.1 headless -c conda-forge -c aihabitat
    /venv/habitat/bin/pip install -q "numpy<2" "datasets>=4" opencv-python-headless scipy matplotlib pyyaml huggingface_hub
    gcc -shared -fPIC -O2 -o tools/egl_software_only.so tools/egl_software_only.c -ldl   # only used with --cpu
    /venv/habitat/bin/python -c "import habitat_sim; print('  habitat-sim', habitat_sim.__version__)"
fi

cat <<'MSG'
== next steps
  evaluate:  python scripts/evaluate.py --repo Jinyan0924/qwen_robotics_open_dataset --config coda_2hz --baseline expert
  train:     python -m vla.train --run runs/<name> --hub_repo <user>/<model-repo> --resume auto ...
             (--hub_repo mirrors every checkpoint to the Hub; --resume auto continues from it on any machine)
  generate:  see "Generate simulated scenarios (Habitat)" in README.md; run scripts/sync_parts_hf.py alongside
MSG
