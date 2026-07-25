"""Compact tables and reports for the complete reviewer candidate matrix."""

from __future__ import annotations

import hashlib
from html import escape
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.chemprop import chemprop_candidates
from opioid.benchmark.context import DEFAULT_OUTPUT_ROOT, EVALUATION_PROTOCOLS
from opioid.benchmark.display_names import add_model_display_columns
from opioid.benchmark.plotting import (
    CV_METRICS,
    PROTOCOL_LABELS,
    STAGE_LABELS,
    benchmark_dir,
    validation_candidate_order,
    write_selected_figures,
)
from opioid.benchmark.reporting import IDENTITY_COLUMNS, METRIC_KEYS


EXPECTED_CANDIDATE_IDS = frozenset(
    candidate.candidate_id for candidate in (*traditional_candidates(), *chemprop_candidates())
)


@dataclass(frozen=True)
class ProtocolResult:
    """Validated summary, fold metrics, and provenance for one protocol."""

    protocol: str
    summary: pd.DataFrame
    matrix_test: pd.DataFrame
    manifest: dict[str, Any]


def write_result_package(
    input_root: Path | str = DEFAULT_OUTPUT_ROOT,
    destination_root: Path | str | None = None,
) -> dict[str, Path]:
    """Write the complete compact figure, table, Markdown, HTML, and hash package."""

    root = Path(input_root)
    destination = root if destination_root is None else Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    figures_dir = destination / "figures"
    results = [_load_protocol_result(protocol, root) for protocol in EVALUATION_PROTOCOLS]
    frames = {result.protocol: add_model_display_columns(result.summary) for result in results}
    order = validation_candidate_order(frames)

    all_candidates = _all_candidates(results, order)
    metrics_long = _metrics_long(all_candidates)
    gaps = _validation_test_gaps(metrics_long)
    outputs: dict[str, Path] = {
        "all_candidates": destination / "all_candidates.csv",
        "metrics_long": destination / "metrics_long.csv",
        "validation_test_gap": destination / "validation_test_gap.csv",
        "report_markdown": destination / "report.md",
        "report_html": destination / "index.html",
        "manifest": destination / "manifest.json",
    }
    all_candidates.to_csv(outputs["all_candidates"], index=False)
    metrics_long.to_csv(outputs["metrics_long"], index=False)
    gaps.to_csv(outputs["validation_test_gap"], index=False)

    figures = write_selected_figures(frames, figures_dir)
    summary = _overall_summary(metrics_long)
    generated_at = datetime.now(timezone.utc).isoformat()
    outputs["report_markdown"].write_text(
        _build_markdown(summary, figures, generated_at), encoding="utf-8"
    )
    outputs["report_html"].write_text(_build_html(summary, figures, generated_at), encoding="utf-8")
    outputs["manifest"].write_text(
        json.dumps(
            _build_manifest(root, destination, results, generated_at),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**outputs, **{f"figure_{name}": path for name, path in figures.items()}}


def write_model_comparison_report(
    input_root: Path | str = DEFAULT_OUTPUT_ROOT,
    destination_root: Path | str | None = None,
) -> dict[str, Path]:
    """Compatibility wrapper for callers of the former report writer."""

    return write_result_package(input_root, destination_root)


def _load_protocol_result(protocol: str, output_root: Path) -> ProtocolResult:
    root = benchmark_dir(protocol, output_root)
    paths = {
        "summary": root / "cv_summary_metrics.csv",
        "matrix_test": root / "matrix_test_metrics.csv",
        "manifest": root / "run_manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing matrix-comparison artifact for {protocol}: {path}")
    summary = pd.read_csv(paths["summary"])
    matrix_test = pd.read_csv(paths["matrix_test"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    _validate_summary(protocol, summary)
    _validate_matrix_test(protocol, matrix_test)
    _validate_manifest(protocol, manifest)
    return ProtocolResult(protocol, summary, matrix_test, manifest)


def _validate_summary(protocol: str, summary: pd.DataFrame) -> None:
    required = {"status", "n_folds", "test_n_folds", *IDENTITY_COLUMNS}
    required.update(
        f"{stage}_{metric}_{stat}"
        for stage in ("valid", "test")
        for metric in METRIC_KEYS
        for stat in ("mean", "std", "ci95_low", "ci95_high")
    )
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"{protocol} summary is missing required columns: {missing}")
    _validate_candidate_ids(protocol, summary, "summary")
    if set(summary["protocol"].astype(str)) != {protocol}:
        raise ValueError(f"{protocol} summary contains mixed or incorrect protocol values")
    if not summary["status"].astype(str).str.casefold().eq("complete").all():
        raise ValueError(f"{protocol} summary contains incomplete candidate results")
    for column in ("n_folds", "test_n_folds"):
        if not pd.to_numeric(summary[column], errors="coerce").eq(5).all():
            raise ValueError(f"{protocol} summary requires {column}=5 for every candidate")
    metric_columns = sorted(required.difference({"status", *IDENTITY_COLUMNS}))
    values = summary[metric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{protocol} summary has non-finite model-comparison metrics")


def _validate_matrix_test(protocol: str, matrix_test: pd.DataFrame) -> None:
    required = {"fold", *IDENTITY_COLUMNS, *METRIC_KEYS}
    missing = sorted(required.difference(matrix_test.columns))
    if missing:
        raise ValueError(f"{protocol} matrix-test table is missing required columns: {missing}")
    _validate_candidate_ids(protocol, matrix_test, "matrix-test")
    grouped = matrix_test.groupby("candidate_id", sort=False)["fold"].agg(list)
    invalid = [
        candidate_id
        for candidate_id, folds in grouped.items()
        if sorted(int(fold) for fold in folds) != list(range(5))
    ]
    if invalid:
        raise ValueError(f"{protocol} matrix-test table has invalid folds: {sorted(invalid)}")
    values = matrix_test[list(METRIC_KEYS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{protocol} matrix-test table has non-finite metrics")


def _validate_candidate_ids(protocol: str, frame: pd.DataFrame, artifact: str) -> None:
    candidate_ids = frozenset(frame["candidate_id"].astype(str))
    if candidate_ids != EXPECTED_CANDIDATE_IDS:
        missing = sorted(EXPECTED_CANDIDATE_IDS.difference(candidate_ids))
        extra = sorted(candidate_ids.difference(EXPECTED_CANDIDATE_IDS))
        raise ValueError(f"{protocol} {artifact} catalog differs; missing={missing}, extra={extra}")


def _validate_manifest(protocol: str, manifest: dict[str, Any]) -> None:
    expected = {
        "protocol": protocol,
        "candidate_count": len(EXPECTED_CANDIDATE_IDS),
        "run_stage": "matrix_test",
        "test_snapshot_included": True,
        "candidate_matrix_frozen_before_test": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{protocol} manifest does not describe a frozen all-candidate test run")
    if "top1_candidate_id" in manifest or "selection_rule" in manifest:
        raise ValueError(f"{protocol} manifest incorrectly carries Top-1 selection metadata")


def _all_candidates(results: list[ProtocolResult], order: list[str]) -> pd.DataFrame:
    rank = {candidate_id: index for index, candidate_id in enumerate(order)}
    frame = add_model_display_columns(
        pd.concat([result.summary for result in results], ignore_index=True)
    )
    frame["_display_order"] = frame["candidate_id"].map(rank)
    frame["_protocol_order"] = frame["protocol"].map(
        {protocol: index for index, protocol in enumerate(EVALUATION_PROTOCOLS)}
    )
    return frame.sort_values(["_protocol_order", "_display_order"], kind="stable").drop(
        columns=["_protocol_order", "_display_order"]
    )


def _metrics_long(all_candidates: pd.DataFrame) -> pd.DataFrame:
    identity = ["protocol", "candidate_id", "model_name", "model_type"]
    rows: list[dict[str, object]] = []
    for row in all_candidates.to_dict(orient="records"):
        for stage in ("valid", "test"):
            for metric, metric_label in CV_METRICS.items():
                rows.append(
                    {
                        **{column: row[column] for column in identity},
                        "stage": stage,
                        "metric": metric_label,
                        "mean": row[f"{stage}_{metric}_mean"],
                        "std": row[f"{stage}_{metric}_std"],
                        "ci95_low": row[f"{stage}_{metric}_ci95_low"],
                        "ci95_high": row[f"{stage}_{metric}_ci95_high"],
                    }
                )
    return pd.DataFrame(rows)


def _validation_test_gaps(metrics_long: pd.DataFrame) -> pd.DataFrame:
    identity = ["protocol", "candidate_id", "model_name", "model_type", "metric"]
    pivot = metrics_long.pivot(index=identity, columns="stage", values="mean").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"valid": "validation_mean", "test": "test_mean"})
    pivot["test_minus_validation"] = pivot["test_mean"] - pivot["validation_mean"]
    return pivot


def _overall_summary(metrics_long: pd.DataFrame) -> pd.DataFrame:
    return metrics_long.groupby(["protocol", "stage", "metric"], as_index=False)["mean"].agg(
        candidate_count="count",
        median="median",
        q1=lambda values: values.quantile(0.25),
        q3=lambda values: values.quantile(0.75),
        minimum="min",
        maximum="max",
    )


def _build_markdown(
    summary: pd.DataFrame,
    figures: dict[str, Path],
    generated_at: str,
) -> str:
    auroc = summary.loc[summary["metric"] == "AUROC"].copy()
    auroc["protocol"] = auroc["protocol"].map(PROTOCOL_LABELS)
    auroc["stage"] = auroc["stage"].map(STAGE_LABELS)
    auroc = auroc.rename(
        columns={
            "protocol": "Protocol",
            "stage": "Stage",
            "candidate_count": "N",
            "median": "Median",
            "q1": "Q1",
            "q3": "Q3",
            "minimum": "Min",
            "maximum": "Max",
        }
    )[["Protocol", "Stage", "N", "Median", "Q1", "Q3", "Min", "Max"]]
    return "\n".join(
        [
            "# μORScreen model comparison",
            "",
            f"Generated at {generated_at}.",
            "",
            "## Evaluation contract",
            "",
            "All 58 frozen candidates are shown with the same naming, ordering, and visual "
            "treatment. Metrics are mean ± sample SD over five fold models; 95% Student-t "
            "intervals are retained in `all_candidates.csv`. Test results are descriptive and "
            "must not drive tuning, threshold changes, ranking, or deployment decisions.",
            "",
            "## AUROC overview",
            "",
            _markdown_table(auroc, decimals=3),
            "",
            "![Validation AUROC](figures/roc_valid.png)",
            "",
            "![Shared-test AUROC](figures/roc_test.png)",
            "",
            "## Files",
            "",
            "- `figures/`: AUROC main figures and metric-level supporting figures.",
            "- `all_candidates.csv`: complete wide-format metrics and provenance fields.",
            "- `metrics_long.csv`: tidy candidate × protocol × stage × metric table.",
            "- `validation_test_gap.csv`: descriptive test-minus-validation differences.",
            "- `manifest.json`: input provenance and output SHA256 checksums.",
            "",
            "No Top-1 model is selected.",
            "",
        ]
    )


def _build_html(summary: pd.DataFrame, figures: dict[str, Path], generated_at: str) -> str:
    auroc = summary.loc[summary["metric"] == "AUROC"].copy()
    auroc["protocol"] = auroc["protocol"].map(PROTOCOL_LABELS)
    auroc["stage"] = auroc["stage"].map(STAGE_LABELS)
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.protocol))}</td><td>{escape(str(row.stage))}</td>"
        f"<td>{int(row.candidate_count)}</td><td>{row.median:.3f}</td>"
        f"<td>{row.q1:.3f}–{row.q3:.3f}</td><td>{row.minimum:.3f}–{row.maximum:.3f}</td>"
        "</tr>"
        for row in auroc.itertuples(index=False)
    )
    gallery = "".join(
        f'<figure><img src="figures/{escape(path.name)}" alt="{escape(name)}">'
        f"<figcaption>{escape(name.replace('_', ' '))}</figcaption></figure>"
        for name, path in figures.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>μORScreen model comparison</title><style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:auto;padding:2rem;color:#20242a}}
h1,h2{{color:#17324d}} .note{{background:#eef4f8;padding:1rem;border-left:4px solid #356d9a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #ccd3d9;padding:.5rem;text-align:left}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}}
figure{{margin:0;border:1px solid #d9dee3;padding:.75rem;background:white}} img{{width:100%;height:auto}}
figcaption{{margin-top:.5rem;text-transform:capitalize;color:#46515b}} code{{background:#f2f4f6;padding:.1rem .3rem}}
</style></head><body><h1>μORScreen model comparison</h1><p>Generated at {escape(generated_at)}.</p>
<div class="note">All 58 frozen candidates use the same visual treatment. Test results are descriptive and must not drive tuning, ranking, threshold changes, or deployment decisions. No Top-1 model is selected.</div>
<h2>AUROC overview</h2><table><thead><tr><th>Protocol</th><th>Stage</th><th>N</th><th>Median</th><th>IQR</th><th>Range</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Figures</h2><div class="gallery">{gallery}</div>
<h2>Data files</h2><ul><li><code>all_candidates.csv</code></li><li><code>metrics_long.csv</code></li><li><code>validation_test_gap.csv</code></li><li><code>manifest.json</code></li></ul>
</body></html>"""


def _build_manifest(
    input_root: Path,
    destination: Path,
    results: list[ProtocolResult],
    generated_at: str,
) -> dict[str, object]:
    files = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    inputs = {}
    for result in results:
        run_manifest = benchmark_dir(result.protocol, input_root) / "run_manifest.json"
        split_files = dict(result.manifest.get("split_snapshot", {}).get("files", {}))
        inputs[result.protocol] = {
            "run_manifest_sha256": _sha256(run_manifest),
            "partition_manifest_sha256": split_files.get("manifest.json"),
        }
    return {
        "kind": "muorscreen.result_package",
        "schema_version": 1,
        "generated_at": generated_at,
        "candidate_count": len(EXPECTED_CANDIDATE_IDS),
        "protocols": list(EVALUATION_PROTOCOLS),
        "ordering": "mean validation AUROC across scaffold, random, and DataSAIL",
        "inputs": inputs,
        "files": files,
    }


def _markdown_table(frame: pd.DataFrame, *, decimals: int | None = None) -> str:
    work = frame.copy()
    if decimals is not None:
        numeric = work.select_dtypes(include=[np.number]).columns
        work[numeric] = work[numeric].round(decimals)
    columns = [str(column) for column in work.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row] for row in work.itertuples(index=False)
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EXPECTED_CANDIDATE_IDS",
    "ProtocolResult",
    "write_model_comparison_report",
    "write_result_package",
]
