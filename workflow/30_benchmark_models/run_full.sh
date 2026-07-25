#!/usr/bin/env bash
set -euo pipefail

workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
results_dir="${workflow_dir}/results"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${results_dir}/_logs/${run_id}"
conda_env="${OPIOID_CONDA_ENV:-opioid_tmp}"

mkdir -p "${log_dir}"

protocols=(scaffold random datasail)
gpus=(0 1 2)
pids=()

terminate_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap terminate_children INT TERM

for index in "${!protocols[@]}"; do
    protocol="${protocols[$index]}"
    gpu="${gpus[$index]}"
    log_path="${log_dir}/${protocol}.log"
    echo "[launch] ${protocol}: physical GPU ${gpu}; log=${log_path}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec conda run --no-capture-output -n "${conda_env}" \
            python "${workflow_dir}/main.py" --protocol "${protocol}"
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
    protocol="${protocols[$index]}"
    if wait "${pids[$index]}"; then
        echo "[complete] ${protocol}"
    else
        echo "[failed] ${protocol}; inspect ${log_dir}/${protocol}.log" >&2
        failed=1
    fi
done

if (( failed != 0 )); then
    exit 1
fi

conda run --no-capture-output -n "${conda_env}" \
    python "${workflow_dir}/validate.py" --write-json
echo "[validated] complete benchmark matrix; logs=${log_dir}"
