"""Adapters for DPNet's canonical split-analysis reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd

from opioid.evaluation.dpbench import DEFAULT_DPNET_COMMAND, PROTOCOLS, TASK_ID


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_ANALYSIS_FILES = (
    "report.html",
    "report.json",
    "analysis_manifest.json",
    "checksums.sha256",
    "split_statistics.csv",
    "label_distribution.csv",
    "molecular_weight_histogram.csv",
    "pairwise_similarity.csv",
    "similarity_histogram.csv",
)
PROTOCOL_LABELS = {
    "scaffold": "Scaffold",
    "random": "Random",
    "datasail": "DataSAIL",
}
PROTOCOL_COLORS = {
    "scaffold": "#4472C4",
    "random": "#A5A5A5",
    "datasail": "#ED7D31",
}
PROXIMITY_X_LABEL = "Maximum Morgan Tanimoto similarity to development set"
PROXIMITY_Y_LABEL = "Fraction of held-out test molecules"


@dataclass(frozen=True)
class SplitAuditResult:
    """Validated paths and summary fields for one canonical DPNet analysis."""

    protocol: str
    output_dir: Path
    report: dict[str, Any]

    @property
    def report_json(self) -> Path:
        return self.output_dir / "report.json"

    @property
    def report_html(self) -> Path:
        return self.output_dir / "report.html"


def run_dpnet_split_audit(
    *,
    protocol: str,
    split_root: Path | str,
    output_dir: Path | str,
    task_id: str = TASK_ID,
    dpnet_executable: str | Sequence[str] = DEFAULT_DPNET_COMMAND,
    replace: bool = False,
) -> SplitAuditResult:
    """Run and validate DPNet's exact, source-read-only split analysis."""

    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    destination = Path(output_dir).resolve()
    command = [
        *_command_prefix(dpnet_executable),
        "analyze",
        task_id,
        "--root",
        str(Path(split_root).resolve()),
        "--processed-dir",
        f"processed_{protocol}",
        "--output",
        str(destination),
    ]
    if replace:
        command.append("--force")
    subprocess.run(command, check=True)
    return load_dpnet_split_audit(
        protocol=protocol,
        output_dir=destination,
        task_id=task_id,
    )


def load_dpnet_split_audit(
    *, protocol: str, output_dir: Path | str, task_id: str = TASK_ID
) -> SplitAuditResult:
    """Load a completed analysis and enforce the μOR reviewer invariants."""

    destination = Path(output_dir)
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DPNet split analysis is incomplete at {destination}; missing={missing}"
        )
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    _validate_report(report, protocol=protocol, task_id=task_id)
    return SplitAuditResult(protocol=protocol, output_dir=destination, report=report)


def write_split_audit_index(
    results: Iterable[SplitAuditResult], output_root: Path | str
) -> dict[str, Path]:
    """Write compact cross-protocol pointers and official evidence metrics."""

    rows = [_summary_row(result) for result in results]
    if {str(row["protocol"]) for row in rows} != set(PROTOCOLS):
        raise ValueError("The split-audit index requires scaffold, random, and datasail reports")
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "dpnet_analysis_summary.csv"
    json_path = destination / "dpnet_analysis_index.json"
    frame = pd.DataFrame(rows).sort_values("protocol").reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "muorscreen.dpnet_split_analysis_index",
                "source": "DPNet dpnet analyze",
                "protocols": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"summary_csv": csv_path, "index_json": json_path}


def write_test_development_proximity_figure(
    results: Iterable[SplitAuditResult],
    output_dir: Path | str,
) -> Path:
    """Plot DPNet's test-to-development nearest-neighbor distributions."""

    by_protocol = {result.protocol: result for result in results}
    if set(by_protocol) != set(PROTOCOLS):
        raise ValueError("The proximity figure requires scaffold, random, and datasail reports")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / "test_to_development_max_tanimoto.png"
    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    ):
        figure, axis = plt.subplots(figsize=(8.2, 5.2))
        for protocol in PROTOCOLS:
            result = by_protocol[protocol]
            report = result.report
            histogram = pd.read_csv(result.output_dir / "similarity_histogram.csv")
            selected = histogram.loc[
                (histogram["kind"] == "nearest_cross_group")
                & (histogram["left_group"] == "development")
                & (histogram["right_group"] == "test")
                & (histogram["direction"] == "right_to_left")
            ].sort_values("bin_lower")
            if selected.empty:
                raise ValueError(f"{protocol}: missing DPNet test-to-development histogram")
            edges = np.r_[
                selected["bin_lower"].to_numpy(dtype=float),
                float(selected["bin_upper"].iloc[-1]),
            ]
            if not np.allclose(
                selected["bin_upper"].to_numpy(dtype=float)[:-1],
                selected["bin_lower"].to_numpy(dtype=float)[1:],
            ):
                raise ValueError(f"{protocol}: non-contiguous DPNet histogram bins")

            groups = {str(item["group"]): item for item in report["split_summary"]}
            count = int(selected["count"].sum())
            expected_count = int(groups["test"]["rows"])
            if count != expected_count:
                raise ValueError(
                    f"{protocol}: histogram contains {count} test molecules, "
                    f"expected {expected_count}"
                )
            pair = next(
                item
                for item in report["pairwise_similarity"]
                if item["left_group"] == "development" and item["right_group"] == "test"
            )
            mean = float(pair["right_nearest_mean"])
            probabilities = selected["count"].to_numpy(dtype=float) / count
            color = PROTOCOL_COLORS[protocol]
            axis.stairs(
                probabilities,
                edges,
                label=f"{PROTOCOL_LABELS[protocol]} (n={count}, mean={mean:.3f})",
                color=color,
                linewidth=2.0,
            )
            axis.axvline(
                mean,
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
            )

        axis.set_xlabel(PROXIMITY_X_LABEL)
        axis.set_ylabel(PROXIMITY_Y_LABEL)
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return figure_path


def _validate_report(report: dict[str, Any], *, protocol: str, task_id: str) -> None:
    if report.get("kind") != "dpnet.split_analysis" or int(report.get("schema_version", 0)) < 2:
        raise ValueError("Unsupported DPNet split-analysis report contract")
    if report.get("task_id") != task_id:
        raise ValueError(f"Split analysis records another task: {report.get('task_id')!r}")
    source = report.get("source")
    if not isinstance(source, dict):
        raise ValueError("Split analysis is missing source metadata")
    expected_dir = f"processed_{protocol}"
    if source.get("processed_dir") != expected_dir:
        raise ValueError(
            f"Split analysis source mismatch: {source.get('processed_dir')!r} != {expected_dir!r}"
        )
    if source.get("method") != protocol or source.get("layout") != "cv":
        raise ValueError(f"Split analysis does not describe the expected {protocol} CV partition")
    similarity = report.get("similarity")
    if not isinstance(similarity, dict) or similarity.get("available") is not True:
        raise ValueError("Fixed-test CV analysis must contain exact cross-group similarity")
    fingerprint = similarity.get("protocol")
    if not isinstance(fingerprint, dict):
        raise ValueError("Split analysis is missing its similarity protocol")
    if (
        str(fingerprint.get("fingerprint", "")).replace(" ", "") != "Morgan/ECFP4"
        or int(fingerprint.get("radius", -1)) != 2
        or int(fingerprint.get("n_bits", -1)) != 1024
        or fingerprint.get("similarity") != "Tanimoto"
    ):
        raise ValueError("DPNet split analysis uses an unexpected molecular similarity contract")
    if protocol == "scaffold":
        overlaps = report.get("scaffolds", {}).get("overlap", [])
        shared = sum(int(item.get("shared_nonempty_scaffolds", 0)) for item in overlaps)
        if shared:
            raise AssertionError(
                f"Scaffold development/test isolation failed in DPNet report: {shared} shared scaffold(s)"
            )


def _summary_row(result: SplitAuditResult) -> dict[str, Any]:
    report = result.report
    groups = {str(item["group"]): item for item in report["split_summary"]}
    pair = report["pairwise_similarity"][0]
    overlaps = report.get("scaffolds", {}).get("overlap", [])
    shared_scaffolds = sum(int(item.get("shared_nonempty_scaffolds", 0)) for item in overlaps)
    leakage = report["similarity"]["datasail_compatible_leakage"]
    return {
        "protocol": result.protocol,
        "dpnet_version": report.get("processor", {}).get("package_version"),
        "n_development": groups["development"]["rows"],
        "n_test": groups["test"]["rows"],
        "shared_nonempty_scaffolds": shared_scaffolds,
        "pair_count": pair["pair_count"],
        "pair_tanimoto_mean": pair["mean"],
        "pair_tanimoto_std": pair["std"],
        "pair_tanimoto_min": pair["min"],
        "pair_tanimoto_max": pair["max"],
        "development_nearest_test_mean": pair["left_nearest_mean"],
        "test_nearest_development_mean": pair["right_nearest_mean"],
        "datasail_scaled_dl": leakage["scaled_dl"],
        "datasail_total_dl": leakage["total_dl"],
        "datasail_max_dl": leakage["max_dl"],
        "datasail_adapter_diagnostics": bool(report.get("datasail", {}).get("available")),
        "report_json": str(result.report_json),
        "report_html": str(result.report_html),
    }


def _command_prefix(executable: str | Sequence[str]) -> list[str]:
    command = shlex.split(executable) if isinstance(executable, str) else list(executable)
    if not command:
        raise ValueError("DPNet executable command must not be empty")
    return command


__all__ = [
    "SplitAuditResult",
    "load_dpnet_split_audit",
    "run_dpnet_split_audit",
    "write_split_audit_index",
    "write_test_development_proximity_figure",
]
