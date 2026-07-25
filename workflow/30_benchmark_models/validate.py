#!/usr/bin/env python3
"""Validate the complete three-protocol reviewer benchmark matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opioid.benchmark.context import EVALUATION_PROTOCOLS
from opioid.benchmark.validation import validate_complete_benchmark


WORKFLOW_DIR = Path(__file__).resolve().parent
SPLIT_ROOT = WORKFLOW_DIR.parent / "10_build_data" / "data" / "dpbench"
OUTPUT_ROOT = WORKFLOW_DIR / "results"
DEFAULT_GPU_MAP = {"scaffold": "0", "random": "1", "datasail": "2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all 58 candidates, five folds, shared tests, and provenance."
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write results/validation_summary.json after all checks pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_complete_benchmark(
        OUTPUT_ROOT,
        split_root=SPLIT_ROOT,
        protocols=EVALUATION_PROTOCOLS,
        expected_gpus=DEFAULT_GPU_MAP,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.write_json:
        output_path = OUTPUT_ROOT / "validation_summary.json"
        output_path.write_text(f"{payload}\n", encoding="utf-8")
        print(output_path)
    else:
        print(payload)


if __name__ == "__main__":
    main()
