#!/usr/bin/env python3
"""Run the Workflow 70 exploratory external-case evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import uuid

from opioid.screen.external_validation import write_external_validation_package


WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parents[1]
TASK_ROOT = (
    WORKFLOW_DIR.parent / "10_build_data" / "data" / "dpbench" / "task_pool" / "muor_antagonism"
)
SOURCE_PATH = WORKFLOW_DIR / "data" / "external_mor_20.csv"
OUTPUT_ROOT = WORKFLOW_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and score the fixed 20-case literature set with the Workflow 60 "
            "LightGBM/TabPFN/RF screening ensemble."
        )
    )
    parser.add_argument("command", choices=("all",))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace this workflow's existing result bundle.",
    )
    return parser.parse_args()


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Workflow 70 requires the opioid_tmp environment with PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Workflow 70 requires a visible CUDA GPU for the Workflow 60 TabPFN artifact"
        )
    print(
        f"[gpu] CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}; "
        f"TabPFN device=cuda:0 ({torch.cuda.get_device_name(0)})",
        flush=True,
    )


def _atomic_build(*, replace: bool) -> None:
    if OUTPUT_ROOT.exists() and not replace:
        raise FileExistsError(f"Results already exist: {OUTPUT_ROOT}. Re-run with --replace.")
    _require_cuda()
    staging = Path(tempfile.mkdtemp(prefix=".70-external-validation-", dir=WORKFLOW_DIR))
    backup = WORKFLOW_DIR / f".70-external-validation-backup-{uuid.uuid4().hex}"
    try:
        bundle = write_external_validation_package(
            repo_root=REPO_ROOT,
            task_root=TASK_ROOT,
            source_path=SOURCE_PATH,
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
    print(f"predictions: {OUTPUT_ROOT / bundle.predictions_path.relative_to(bundle.root)}")
    print(f"metrics: {OUTPUT_ROOT / bundle.metrics_path.relative_to(bundle.root)}")
    print(f"report: {OUTPUT_ROOT / bundle.report_path.relative_to(bundle.root)}")
    print(f"manifest: {OUTPUT_ROOT / bundle.manifest_path.relative_to(bundle.root)}")


def main() -> None:
    args = parse_args()
    _atomic_build(replace=args.replace)


if __name__ == "__main__":
    main()
