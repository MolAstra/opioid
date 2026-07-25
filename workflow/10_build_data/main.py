#!/usr/bin/env python3

"""Build the DPBench local 5CV + fixed-test partitions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from opioid.evaluation.dpbench import (
    DEFAULT_DPNET_COMMAND,
    PROTOCOLS,
    TASK_ID,
    build_partitions,
    prepare_task,
    validate_partitions,
    write_runtime_snapshot,
)

WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = WORKFLOW_DIR / "data" / "source" / "opioid.csv"
DEFAULT_DPBENCH_ROOT = WORKFLOW_DIR / "data" / "dpbench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPBench scaffold, random, and DataSAIL 5CV + fixed-test partitions."
    )
    parser.add_argument(
        "--protocol",
        choices=("all", *PROTOCOLS),
        default="all",
        help="Protocol to rebuild; default: all.",
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--dpbench-root",
        type=Path,
        default=DEFAULT_DPBENCH_ROOT,
        help="DPBench task-pool root (default: this workflow's data/dpbench).",
    )
    parser.add_argument(
        "--datasail-source",
        type=Path,
        default=None,
        help="Optional DataSAIL Git checkout to include in the runtime snapshot.",
    )
    parser.add_argument(
        "--dpnet",
        default=DEFAULT_DPNET_COMMAND,
        help="Command used to invoke dpnet (default runs dpnet in conda env molm).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace selected local partitions; changed input requires all protocols.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    paths = prepare_task(
        input_csv=args.input_csv,
        root=args.dpbench_root,
        task_id=TASK_ID,
        protocols=protocols,
        replace=args.replace,
    )
    build_partitions(
        root=paths.root,
        task_id=TASK_ID,
        protocols=protocols,
        dpnet_executable=args.dpnet,
        replace=args.replace,
    )
    validate_partitions(
        root=paths.root,
        task_id=TASK_ID,
        protocols=protocols,
        dpnet_executable=args.dpnet,
    )
    snapshot = write_runtime_snapshot(
        root=paths.root,
        task_id=TASK_ID,
        dpnet_executable=args.dpnet,
        datasail_source=args.datasail_source,
    )
    print(f"DPBench task root: {paths.task_root}")
    for protocol in protocols:
        print(f"[{protocol}] 5CV + fixed test: {paths.processed_dir(protocol)}")
    print(f"DPBench/DataSAIL runtime snapshot: {snapshot}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "molm":
        print("note: datasail requires the DPBench/DataSAIL-enabled molm environment")


if __name__ == "__main__":
    main()
