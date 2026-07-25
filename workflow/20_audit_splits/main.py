#!/usr/bin/env python3
"""Audit DPBench development/test and train/validation similarity distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from opioid.evaluation.dpbench import DEFAULT_DPNET_COMMAND, PROTOCOLS, TASK_ID
from opioid.evaluation.similarity import (
    diagnose_cv_folds,
    write_cv_fold_similarity_diagnostics,
)
from opioid.evaluation.split_audit import (
    run_dpnet_split_audit,
    write_split_audit_index,
    write_test_development_proximity_figure,
)


WORKFLOW_DIR = Path(__file__).resolve().parent
SPLIT_ROOT = WORKFLOW_DIR.parent / "10_build_data" / "data" / "dpbench"
RESULTS_ROOT = WORKFLOW_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical DPNet split analysis and supplemental per-fold Morgan diagnostics."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=(*PROTOCOLS, "all"),
        default="all",
        help="DPBench protocol to audit; default: all.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SPLIT_ROOT,
        help="DPBench task-pool root (default: workflow/10_build_data/data/dpbench).",
    )
    parser.add_argument(
        "--dpnet",
        default=DEFAULT_DPNET_COMMAND,
        help="Command used for the canonical dpnet analyze report.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace existing DPNet analysis report directories.",
    )
    parser.add_argument(
        "--no-fold-diagnostics",
        action="store_true",
        help="Write only canonical DPNet reports, without supplemental train/valid diagnostics.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root for per-protocol diagnostics (default: this workflow's results).",
    )
    return parser.parse_args()


def main(
    protocol: str = "all",
    split_root: Path = SPLIT_ROOT,
    output_root: Path = RESULTS_ROOT,
    dpnet: str = DEFAULT_DPNET_COMMAND,
    replace: bool = False,
    fold_diagnostics: bool = True,
) -> dict[str, dict[str, Path]]:
    """Generate canonical DPNet reports plus optional per-fold diagnostics."""

    selected = PROTOCOLS if protocol == "all" else (protocol,)
    all_outputs: dict[str, dict[str, Path]] = {}
    official_results = []
    for name in selected:
        split_dir = split_root / "task_pool" / TASK_ID / f"processed_{name}"
        target_dir = output_root / name
        official = run_dpnet_split_audit(
            protocol=name,
            split_root=split_root,
            output_dir=target_dir / "dpnet_analysis",
            dpnet_executable=dpnet,
            replace=replace,
        )
        official_results.append(official)
        outputs = {
            "dpnet_report_html": official.report_html,
            "dpnet_report_json": official.report_json,
        }
        print(f"[{name}] canonical DPNet analysis: {official.report_html}")
        if fold_diagnostics:
            result = diagnose_cv_folds(split_dir, protocol=name)
            supplemental = write_cv_fold_similarity_diagnostics(
                result, target_dir / "fold_diagnostics"
            )
            outputs.update({f"fold_{label}": path for label, path in supplemental.items()})
            print(
                f"[{name}] supplemental CV diagnostics: {len(result.folds)} folds; "
                "fixed test not read"
            )
        for label, path in outputs.items():
            print(f"saved {label}: {path}")
        all_outputs[name] = outputs
    if protocol == "all":
        for label, path in write_split_audit_index(official_results, output_root).items():
            print(f"saved {label}: {path}")
        proximity_figure = write_test_development_proximity_figure(
            official_results,
            output_root / "figures",
        )
        all_outputs["cross_protocol"] = {
            "test_to_development_max_tanimoto": proximity_figure
        }
        print(f"saved cross-protocol proximity figure: {proximity_figure}")
    return all_outputs


if __name__ == "__main__":
    args = parse_args()
    main(
        args.protocol,
        args.split_root,
        args.output_root,
        args.dpnet,
        args.replace,
        not args.no_fold_diagnostics,
    )
