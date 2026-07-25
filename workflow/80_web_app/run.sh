#!/usr/bin/env bash
set -euo pipefail

workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${workflow_dir}/run_server.sh" "$@"
