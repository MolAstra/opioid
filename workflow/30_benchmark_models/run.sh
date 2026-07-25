#!/usr/bin/env bash
set -euo pipefail

workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "${workflow_dir}/main.py" "$@"
