"""Normalized CSV reporting for the frozen reviewer candidate matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_KEYS = ("roc", "prc", "accuracy", "f1", "mcc")
IDENTITY_COLUMNS = ("protocol", "family", "algorithm", "preset", "representation", "candidate_id")
_T95_N5 = 2.7764451051977987


def summarize_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five-fold validation metrics without selecting a candidate."""

    if fold_metrics.empty:
        raise ValueError("Cannot summarize an empty CV result table")
    required = set(IDENTITY_COLUMNS).union({"fold", *(f"valid_{key}" for key in METRIC_KEYS)})
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"Fold metrics are missing required columns: {sorted(missing)}")

    aggregation: dict[str, tuple[str, str]] = {"n_folds": ("fold", "count")}
    for metric in METRIC_KEYS:
        aggregation[f"valid_{metric}_mean"] = (f"valid_{metric}", "mean")
        aggregation[f"valid_{metric}_std"] = (f"valid_{metric}", "std")
    summary = fold_metrics.groupby(list(IDENTITY_COLUMNS), as_index=False).agg(**aggregation)
    summary["status"] = "complete"
    return _add_ci95(summary, prefix="valid")


def merge_matrix_test(summary: pd.DataFrame, matrix_test: pd.DataFrame | None) -> pd.DataFrame:
    """Attach five shared-test fold scores for every frozen candidate."""

    work = summary.copy()
    if matrix_test is None or matrix_test.empty:
        work["test_n_folds"] = pd.NA
        for metric in METRIC_KEYS:
            work[f"test_{metric}_mean"] = np.nan
            work[f"test_{metric}_std"] = np.nan
            work[f"test_{metric}_ci95_low"] = np.nan
            work[f"test_{metric}_ci95_high"] = np.nan
        return work

    required = set(IDENTITY_COLUMNS).union({"fold", *METRIC_KEYS})
    missing = required.difference(matrix_test.columns)
    if missing:
        raise ValueError(f"Matrix test metrics are missing required columns: {sorted(missing)}")
    candidate_ids = set(matrix_test["candidate_id"].astype(str))
    summary_ids = set(work["candidate_id"].astype(str))
    if candidate_ids != summary_ids:
        raise ValueError("Matrix test candidate set differs from the validation summary")
    fold_counts = matrix_test.groupby("candidate_id", sort=False)["fold"].agg(list)
    invalid_candidates = [
        candidate_id
        for candidate_id, folds in fold_counts.items()
        if sorted(int(fold) for fold in folds) != list(range(5))
    ]
    if invalid_candidates:
        raise ValueError(
            "Every matrix-test candidate must contain exactly folds 0 through 4; "
            f"invalid={sorted(invalid_candidates)}"
        )
    aggregation: dict[str, tuple[str, str]] = {"test_n_folds": ("fold", "count")}
    for metric in METRIC_KEYS:
        aggregation[f"test_{metric}_mean"] = (metric, "mean")
        aggregation[f"test_{metric}_std"] = (metric, "std")
    aggregate = matrix_test.groupby(list(IDENTITY_COLUMNS), as_index=False).agg(**aggregation)
    aggregate["n_folds"] = aggregate["test_n_folds"]
    aggregate = _add_ci95(aggregate, prefix="test").drop(columns="n_folds")
    return work.merge(aggregate, on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")


def _add_ci95(summary: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Add a Student-t 95% interval for the fixed five-fold protocol."""

    work = summary.copy()
    for metric in METRIC_KEYS:
        mean = work[f"{prefix}_{metric}_mean"].astype(float)
        std = work[f"{prefix}_{metric}_std"].astype(float)
        n_column = "n_folds" if prefix == "valid" else "test_n_folds"
        n = work[n_column].astype(float)
        margin = _T95_N5 * std / np.sqrt(n)
        work[f"{prefix}_{metric}_ci95_low"] = mean - margin
        work[f"{prefix}_{metric}_ci95_high"] = mean + margin
    return work


def write_tables(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    matrix_test: pd.DataFrame | None,
) -> dict[str, Path]:
    """Write stable reviewer-facing tables for one protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "cv_summary_metrics.csv"
    matrix_test_path = output_dir / "matrix_test_metrics.csv"
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    if matrix_test is None:
        pd.DataFrame(
            columns=[*IDENTITY_COLUMNS, *METRIC_KEYS, "artifact_type", "artifact_path"]
        ).to_csv(matrix_test_path, index=False)
    else:
        matrix_test.to_csv(matrix_test_path, index=False)
    return {"fold_metrics": fold_path, "summary": summary_path, "matrix_test": matrix_test_path}
