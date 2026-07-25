#!/usr/bin/env python3
"""Build the full-corpus RF explanation and screening-model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import uuid

from opioid.explain.full_rf import write_explanation_package


WORKFLOW_DIR = Path(__file__).resolve().parent
TASK_ROOT = (
    WORKFLOW_DIR.parent / "10_build_data" / "data" / "dpbench" / "task_pool" / "muor_antagonism"
)
BENCHMARK_ROOT = WORKFLOW_DIR.parent / "30_benchmark_models" / "results"
REPORT_ROOT = WORKFLOW_DIR.parent / "40_report_results" / "results"
OUTPUT_ROOT = WORKFLOW_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RF/ECFP4-2048 on all 982 deduplicated labels and generate the core "
            "TreeSHAP/SAR explanation package."
        )
    )
    parser.add_argument("command", choices=("all",))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace this workflow's existing result bundle.",
    )
    return parser.parse_args()


def _atomic_build(*, replace: bool) -> None:
    if OUTPUT_ROOT.exists() and not replace:
        raise FileExistsError(f"Results already exist: {OUTPUT_ROOT}. Re-run with --replace.")
    staging = Path(tempfile.mkdtemp(prefix=".50-explain-rf-", dir=WORKFLOW_DIR))
    backup = WORKFLOW_DIR / f".50-explain-rf-backup-{uuid.uuid4().hex}"
    try:
        bundle = write_explanation_package(
            task_root=TASK_ROOT,
            benchmark_root=BENCHMARK_ROOT,
            report_root=REPORT_ROOT,
            destination=staging,
        )
        if OUTPUT_ROOT.exists():
            OUTPUT_ROOT.rename(backup)
        staging.rename(OUTPUT_ROOT)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if not OUTPUT_ROOT.exists() and backup.exists():
            backup.rename(OUTPUT_ROOT)
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"model: {OUTPUT_ROOT / bundle.model_path.relative_to(bundle.root)}")
    print(f"report: {OUTPUT_ROOT / bundle.report_path.relative_to(bundle.root)}")
    print(f"manifest: {OUTPUT_ROOT / bundle.manifest_path.relative_to(bundle.root)}")


def main() -> None:
    args = parse_args()
    _atomic_build(replace=args.replace)


if __name__ == "__main__":
    main()
