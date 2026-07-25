#!/usr/bin/env bash
set -euo pipefail

workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_env="${OPIOID_CONDA_ENV:-opioid_tmp}"
physical_gpu="${OPIOID_WEB_GPU:-5}"
host="${OPIOID_WEB_HOST:-127.0.0.1}"
port="${OPIOID_WEB_PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
echo "[launch] Workflow 80: http://${host}:${port}; physical GPU ${physical_gpu}; env ${conda_env}"
exec conda run --no-capture-output -n "${conda_env}" \
    python "${workflow_dir}/main.py" --host "${host}" --port "${port}" "$@"
