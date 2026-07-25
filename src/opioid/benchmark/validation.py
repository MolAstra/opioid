"""Strict acceptance checks for complete reviewer benchmark result matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.chemprop import chemprop_candidates
from opioid.benchmark.context import EVALUATION_PROTOCOLS, RunContext
from opioid.benchmark.reporting import IDENTITY_COLUMNS, METRIC_KEYS


EXPECTED_CANDIDATE_IDS = frozenset(
    candidate.candidate_id for candidate in (*traditional_candidates(), *chemprop_candidates())
)
EXPECTED_FAMILY_COUNTS = {"traditional": 55, "chemprop": 3}
EXPECTED_FOLDS = list(range(5))


def validate_complete_benchmark(
    output_root: Path | str,
    *,
    split_root: Path | str,
    protocols: tuple[str, ...] = EVALUATION_PROTOCOLS,
    expected_gpus: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate all tables, manifests, split hashes, and GPU provenance."""

    root = Path(output_root)
    split_root = Path(split_root)
    summaries = [
        _validate_protocol(
            protocol,
            root,
            split_root,
            expected_gpu=None if expected_gpus is None else expected_gpus.get(protocol),
        )
        for protocol in protocols
    ]
    return {
        "kind": "muorscreen.complete_benchmark_validation",
        "schema_version": 1,
        "valid": True,
        "output_root": str(root.resolve()),
        "protocols": summaries,
    }


def _validate_protocol(
    protocol: str,
    output_root: Path,
    split_root: Path,
    *,
    expected_gpu: str | None,
) -> dict[str, Any]:
    context = RunContext.for_protocol(
        protocol,
        split_root=split_root,
        output_root=output_root,
    )
    benchmark = context.output_dir
    paths = {
        "summary": benchmark / "cv_summary_metrics.csv",
        "fold_metrics": benchmark / "fold_metrics.csv",
        "matrix_test": benchmark / "matrix_test_metrics.csv",
        "manifest": benchmark / "run_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{protocol} is missing benchmark artifacts: {missing}")

    summary = pd.read_csv(paths["summary"])
    fold_metrics = pd.read_csv(paths["fold_metrics"])
    matrix_test = pd.read_csv(paths["matrix_test"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    _validate_candidate_table(protocol, summary, expected_rows=58, table_name="summary")
    _validate_candidate_table(protocol, fold_metrics, expected_rows=290, table_name="fold metrics")
    _validate_candidate_table(protocol, matrix_test, expected_rows=290, table_name="matrix test")
    _validate_folds(protocol, fold_metrics, "fold metrics")
    _validate_folds(protocol, matrix_test, "matrix test")

    if not summary["status"].astype(str).str.casefold().eq("complete").all():
        raise ValueError(f"{protocol} summary contains incomplete candidates")
    for column in ("n_folds", "test_n_folds"):
        if not pd.to_numeric(summary[column], errors="coerce").eq(5).all():
            raise ValueError(f"{protocol} summary requires {column}=5 for every candidate")

    summary_metrics = [
        f"{prefix}_{metric}_{stat}"
        for prefix in ("valid", "test")
        for metric in METRIC_KEYS
        for stat in ("mean", "std", "ci95_low", "ci95_high")
    ]
    _require_finite(protocol, summary, summary_metrics, "summary")
    _require_finite(
        protocol,
        fold_metrics,
        [f"valid_{metric}" for metric in METRIC_KEYS],
        "fold metrics",
    )
    _require_finite(protocol, matrix_test, list(METRIC_KEYS), "matrix test")

    _validate_manifest(protocol, manifest, context, expected_gpu=expected_gpu)
    manifest_hash = _sha256(context.split_dir / "manifest.json")
    recorded_hash = dict(manifest["split_snapshot"]["files"]).get("manifest.json")
    if recorded_hash != manifest_hash:
        raise ValueError(
            f"{protocol} run manifest split hash differs from the persisted partition manifest"
        )
    return {
        "protocol": protocol,
        "candidate_count": len(summary),
        "validation_rows": len(fold_metrics),
        "shared_test_rows": len(matrix_test),
        "family_counts": summary["family"].value_counts().sort_index().to_dict(),
        "split_manifest_sha256": manifest_hash,
        "cuda_visible_devices": dict(manifest["chemprop_runtime"])["cuda_visible_devices"],
        "run_manifest": str(paths["manifest"].resolve()),
    }


def _validate_candidate_table(
    protocol: str,
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    table_name: str,
) -> None:
    required = {"candidate_id", "family", *IDENTITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{protocol} {table_name} is missing columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(f"{protocol} {table_name} has {len(frame)} rows, expected {expected_rows}")
    candidate_ids = frozenset(frame["candidate_id"].astype(str))
    if candidate_ids != EXPECTED_CANDIDATE_IDS:
        missing_ids = sorted(EXPECTED_CANDIDATE_IDS.difference(candidate_ids))
        extra_ids = sorted(candidate_ids.difference(EXPECTED_CANDIDATE_IDS))
        raise ValueError(
            f"{protocol} {table_name} candidate catalog differs; "
            f"missing={missing_ids}, extra={extra_ids}"
        )
    family_counts = frame.drop_duplicates("candidate_id")["family"].value_counts().to_dict()
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"{protocol} {table_name} family counts differ: {family_counts}")


def _validate_folds(protocol: str, frame: pd.DataFrame, table_name: str) -> None:
    if "fold" not in frame:
        raise ValueError(f"{protocol} {table_name} is missing fold")
    grouped = frame.groupby("candidate_id", sort=False)["fold"].agg(list)
    invalid = [
        candidate_id
        for candidate_id, folds in grouped.items()
        if sorted(int(fold) for fold in folds) != EXPECTED_FOLDS
    ]
    if invalid:
        raise ValueError(f"{protocol} {table_name} has invalid folds: {sorted(invalid)}")


def _require_finite(
    protocol: str,
    frame: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{protocol} {table_name} is missing metric columns: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{protocol} {table_name} contains non-finite metrics")


def _validate_manifest(
    protocol: str,
    manifest: dict[str, Any],
    context: RunContext,
    *,
    expected_gpu: str | None,
) -> None:
    required_values = {
        "protocol": protocol,
        "candidate_count": 58,
        "run_stage": "matrix_test",
        "test_snapshot_included": True,
        "candidate_matrix_frozen_before_test": True,
    }
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in required_values.items()
        if manifest.get(key) != value
    }
    if mismatched:
        raise ValueError(f"{protocol} run manifest contract differs: {mismatched}")
    if set(manifest.get("suites", [])) != {"traditional", "chemprop"}:
        raise ValueError(f"{protocol} run manifest does not contain both benchmark suites")
    if "top1_candidate_id" in manifest or "selection_rule" in manifest:
        raise ValueError(f"{protocol} run manifest contains forbidden Top-1 metadata")
    runtime = manifest.get("chemprop_runtime")
    if not isinstance(runtime, dict) or runtime.get("accelerator") != "gpu":
        raise ValueError(f"{protocol} run manifest does not record GPU Chemprop")
    visible = runtime.get("cuda_visible_devices")
    if expected_gpu is not None and str(visible) != str(expected_gpu):
        raise ValueError(
            f"{protocol} expected CUDA_VISIBLE_DEVICES={expected_gpu}, recorded {visible!r}"
        )
    if Path(manifest.get("split_dir", "")).resolve() != context.split_dir.resolve():
        raise ValueError(f"{protocol} run manifest records an unexpected split directory")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EXPECTED_CANDIDATE_IDS", "validate_complete_benchmark"]
