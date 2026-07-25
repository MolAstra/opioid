from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.chemprop import chemprop_candidates
from opioid.benchmark.context import RunContext
from opioid.benchmark.validation import validate_complete_benchmark


PROTOCOL_GPUS = {"scaffold": "0", "random": "1", "datasail": "2"}
METRICS = ("roc", "prc", "accuracy", "f1", "mcc")


def _metadata(protocol: str, candidate: object) -> dict[str, str]:
    if hasattr(candidate, "algorithm_id"):
        return {
            "protocol": protocol,
            "family": "traditional",
            "algorithm": str(candidate.algorithm_id),
            "preset": str(candidate.algorithm_id),
            "representation": str(candidate.representation_id),
            "candidate_id": str(candidate.candidate_id),
        }
    return {
        "protocol": protocol,
        "family": "chemprop",
        "algorithm": str(candidate.algorithm),
        "preset": str(candidate.preset),
        "representation": str(candidate.representation),
        "candidate_id": str(candidate.candidate_id),
    }


def _write_complete_protocol(output_root: Path, split_root: Path, protocol: str) -> None:
    candidates = (*traditional_candidates(), *chemprop_candidates())
    context = RunContext.for_protocol(protocol, output_root=output_root, split_root=split_root)
    context.output_dir.mkdir(parents=True)
    context.split_dir.mkdir(parents=True)
    split_manifest = context.split_dir / "manifest.json"
    split_manifest.write_text(json.dumps({"protocol": protocol}), encoding="utf-8")
    split_hash = hashlib.sha256(split_manifest.read_bytes()).hexdigest()

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for candidate in candidates:
        metadata = _metadata(protocol, candidate)
        summary: dict[str, object] = {
            **metadata,
            "n_folds": 5,
            "test_n_folds": 5,
            "status": "complete",
        }
        for prefix in ("valid", "test"):
            for metric in METRICS:
                summary[f"{prefix}_{metric}_mean"] = 0.8
                summary[f"{prefix}_{metric}_std"] = 0.02
                summary[f"{prefix}_{metric}_ci95_low"] = 0.77
                summary[f"{prefix}_{metric}_ci95_high"] = 0.83
        summary_rows.append(summary)
        for fold in range(5):
            fold_rows.append(
                {**metadata, "fold": fold, **{f"valid_{metric}": 0.8 for metric in METRICS}}
            )
            matrix_rows.append({**metadata, "fold": fold, **{metric: 0.8 for metric in METRICS}})

    pd.DataFrame(summary_rows).to_csv(context.output_dir / "cv_summary_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(context.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(context.output_dir / "matrix_test_metrics.csv", index=False)
    (context.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "candidate_count": 58,
                "suites": ["traditional", "chemprop"],
                "run_stage": "matrix_test",
                "test_snapshot_included": True,
                "candidate_matrix_frozen_before_test": True,
                "split_dir": str(context.split_dir.resolve()),
                "split_snapshot": {"files": {"manifest.json": split_hash}},
                "chemprop_runtime": {
                    "accelerator": "gpu",
                    "devices": "1",
                    "cuda_visible_devices": PROTOCOL_GPUS[protocol],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_complete_results(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "results"
    split_root = tmp_path / "splits"
    for protocol in PROTOCOL_GPUS:
        _write_complete_protocol(output_root, split_root, protocol)
    return output_root, split_root


def test_complete_benchmark_validator_accepts_exact_matrix(tmp_path: Path) -> None:
    output_root, split_root = _write_complete_results(tmp_path)

    result = validate_complete_benchmark(
        output_root,
        split_root=split_root,
        expected_gpus=PROTOCOL_GPUS,
    )

    assert result["valid"] is True
    assert [item["candidate_count"] for item in result["protocols"]] == [58, 58, 58]
    assert [item["shared_test_rows"] for item in result["protocols"]] == [290, 290, 290]


def test_complete_benchmark_validator_rejects_missing_test_fold(tmp_path: Path) -> None:
    output_root, split_root = _write_complete_results(tmp_path)
    path = output_root / "scaffold" / "benchmark" / "matrix_test_metrics.csv"
    frame = pd.read_csv(path).iloc[:-1]
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="289 rows, expected 290"):
        validate_complete_benchmark(
            output_root,
            split_root=split_root,
            expected_gpus=PROTOCOL_GPUS,
        )
