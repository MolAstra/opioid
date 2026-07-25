#!/usr/bin/env python3
"""Generate the compact, unified reviewer result package."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import uuid

from opioid.benchmark.context import EVALUATION_PROTOCOLS
from opioid.benchmark.plotting import load_protocol_frames, write_selected_figures
from opioid.benchmark.result_report import write_result_package


WORKFLOW_DIR = Path(__file__).resolve().parent
INPUT_ROOT = WORKFLOW_DIR.parent / "30_benchmark_models" / "results"
OUTPUT_ROOT = WORKFLOW_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unified figures, tables, and reports for all frozen candidates."
    )
    parser.add_argument("command", choices=("plot", "report", "all"))
    parser.add_argument(
        "--stage",
        "--metrics",
        choices=("valid", "test", "both"),
        default="both",
        help="Evaluation stage to plot; --metrics is retained as a compatibility alias.",
    )
    parser.add_argument(
        "--protocol",
        choices=(*EVALUATION_PROTOCOLS, "all"),
        default="all",
        help="Protocol overview to plot; cross-split figures require all protocols.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the workflow-owned result package after successful generation.",
    )
    return parser.parse_args()


def _stages(value: str) -> tuple[str, ...]:
    return ("valid", "test") if value == "both" else (value,)


def _plot(protocol: str, stage: str, *, replace: bool) -> None:
    protocols = EVALUATION_PROTOCOLS if protocol == "all" else (protocol,)
    figures_dir = OUTPUT_ROOT / "figures"
    if replace and figures_dir.exists():
        shutil.rmtree(figures_dir)
    frames = load_protocol_frames(INPUT_ROOT, protocols)
    outputs = write_selected_figures(frames, figures_dir, stages=_stages(stage))
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _atomic_package(*, replace: bool) -> None:
    if OUTPUT_ROOT.exists() and not replace:
        raise FileExistsError(
            f"Result package already exists: {OUTPUT_ROOT}. Re-run with --replace."
        )
    staging = Path(tempfile.mkdtemp(prefix=".40-report-", dir=WORKFLOW_DIR))
    backup = WORKFLOW_DIR / f".40-report-backup-{uuid.uuid4().hex}"
    try:
        write_result_package(INPUT_ROOT, staging)
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
    print(f"result package: {OUTPUT_ROOT}")


def main() -> None:
    args = parse_args()
    if args.command == "plot":
        _plot(args.protocol, args.stage, replace=args.replace)
        return
    if args.protocol != "all" or args.stage != "both":
        raise ValueError("report/all require --protocol all and --stage both")
    _atomic_package(replace=args.replace)


if __name__ == "__main__":
    main()
