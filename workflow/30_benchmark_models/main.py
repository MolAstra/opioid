#!/usr/bin/env python3
"""Run the frozen reviewer model matrix without test-driven decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

from opioid.benchmark import (
    EVALUATION_PROTOCOLS,
    RunContext,
    run_all_protocol_benchmarks,
    run_protocol_benchmark,
)
from opioid.evaluation.dpbench import TASK_ID


WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_ROOT = WORKFLOW_DIR.parent / "10_build_data" / "data" / "dpbench"
DEFAULT_OUTPUT_ROOT = WORKFLOW_DIR / "results"


def _progress(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the μORScreen ML×representation and Chemprop reviewer benchmarks. "
            "All frozen candidates receive five-fold CV and shared-test metrics."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=("all", *EVALUATION_PROTOCOLS),
        default="all",
        help="Named evaluation protocol(s); default runs all three isolated reviewer protocols.",
    )
    parser.add_argument(
        "--suite",
        choices=("traditional", "chemprop"),
        action="append",
        dest="suites",
        help="Benchmark suite to run; repeat to select both (default: both).",
    )
    parser.add_argument(
        "--no-test",
        "--no-final-test",
        action="store_true",
        dest="no_test",
        help="Run CV only. This never reads the independent test CSV.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
        help="DPBench root produced by workflow/10_build_data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = tuple(args.suites or ("traditional", "chemprop"))
    if args.protocol == "all":
        results = run_all_protocol_benchmarks(
            protocols=EVALUATION_PROTOCOLS,
            suites=suites,
            evaluate_test=not args.no_test,
            split_root=args.split_root,
            output_root=DEFAULT_OUTPUT_ROOT,
            progress=_progress,
        )
    else:
        results = [
            run_protocol_benchmark(
                RunContext.for_protocol(
                    args.protocol,
                    split_root=args.split_root,
                    output_root=DEFAULT_OUTPUT_ROOT,
                    task_id=TASK_ID,
                ),
                suites=suites,
                evaluate_test=not args.no_test,
                progress=_progress,
            )
        ]

    for result in results:
        print(f"[{result.context.protocol}] fold metrics: {result.fold_metrics_csv}")
        print(f"[{result.context.protocol}] CV summary: {result.summary_csv}")
        print(f"[{result.context.protocol}] matrix test: {result.matrix_test_csv}")
        print(f"[{result.context.protocol}] manifest: {result.run_manifest_json}")


if __name__ == "__main__":
    main()
