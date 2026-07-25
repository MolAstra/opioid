#!/usr/bin/env python3
"""Run the workflow-owned full-data consensus virtual screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import uuid

from opioid.screen.virtual_screen import write_virtual_screen_package


WORKFLOW_DIR = Path(__file__).resolve().parent
TASK_ROOT = (
    WORKFLOW_DIR.parent / "10_build_data" / "data" / "dpbench" / "task_pool" / "muor_antagonism"
)
RF_BUNDLE_ROOT = WORKFLOW_DIR.parent / "50_explain_rf" / "results"
BENCHMARK_REPORT_ROOT = WORKFLOW_DIR.parent / "40_report_results" / "results"
SOURCE_ROOT = WORKFLOW_DIR / "data" / "sources"
OUTPUT_ROOT = WORKFLOW_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the DataSAIL-selected LightGBM and TabPFN models on all labels, reuse the "
            "Workflow 50 RF, and screen four molecular snapshots by three-model consensus."
        )
    )
    parser.add_argument("command", choices=("all",))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace this workflow's existing result bundle.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Canonical molecules per feature/prediction batch (default: 1024).",
    )
    return parser.parse_args()


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Workflow 60 requires the opioid_tmp environment with PyTorch"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Workflow 60 requires a visible CUDA GPU for TabPFN. Use run.sh or set "
            "CUDA_VISIBLE_DEVICES explicitly."
        )
    print(
        f"[gpu] CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}; "
        f"TabPFN device=cuda:0 ({torch.cuda.get_device_name(0)})",
        flush=True,
    )


def _atomic_build(*, replace: bool, batch_size: int) -> None:
    if OUTPUT_ROOT.exists() and not replace:
        raise FileExistsError(f"Results already exist: {OUTPUT_ROOT}. Re-run with --replace.")
    _require_cuda()
    staging = Path(tempfile.mkdtemp(prefix=".60-virtual-screen-", dir=WORKFLOW_DIR))
    backup = WORKFLOW_DIR / f".60-virtual-screen-backup-{uuid.uuid4().hex}"
    try:
        bundle = write_virtual_screen_package(
            task_root=TASK_ROOT,
            rf_bundle_root=RF_BUNDLE_ROOT,
            source_root=SOURCE_ROOT,
            benchmark_report_root=BENCHMARK_REPORT_ROOT,
            destination=staging,
            batch_size=batch_size,
            progress=lambda message: print(message, flush=True),
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
    print(f"report: {OUTPUT_ROOT / bundle.report_path.relative_to(bundle.root)}")
    print(f"manifest: {OUTPUT_ROOT / bundle.manifest_path.relative_to(bundle.root)}")
    print(f"consensus: {OUTPUT_ROOT / bundle.consensus_path.relative_to(bundle.root)}")


def main() -> None:
    args = parse_args()
    _atomic_build(replace=args.replace, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
