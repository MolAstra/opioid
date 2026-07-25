from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.chemprop import chemprop_candidates
from opioid.benchmark import result_report
from opioid.benchmark.result_report import write_model_comparison_report


def _candidates():
    return [*traditional_candidates(), *chemprop_candidates()]


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


def _write_complete_protocol(output_root: Path, protocol: str) -> None:
    benchmark = output_root / protocol / "benchmark"
    benchmark.mkdir(parents=True)
    summary_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(_candidates()):
        metadata = _metadata(protocol, candidate)
        row: dict[str, object] = {**metadata, "n_folds": 5, "test_n_folds": 5, "status": "complete"}
        for metric in ("roc", "prc", "accuracy", "f1", "mcc"):
            row[f"valid_{metric}_mean"] = 0.80 - rank / 1000
            row[f"valid_{metric}_std"] = 0.02
            row[f"valid_{metric}_ci95_low"] = 0.775 - rank / 1000
            row[f"valid_{metric}_ci95_high"] = 0.825 - rank / 1000
            row[f"test_{metric}_mean"] = 0.78 - rank / 1000
            row[f"test_{metric}_std"] = 0.03
            row[f"test_{metric}_ci95_low"] = 0.74 - rank / 1000
            row[f"test_{metric}_ci95_high"] = 0.82 - rank / 1000
        summary_rows.append(row)
        for fold in range(5):
            matrix_rows.append(
                {
                    **metadata,
                    "fold": fold,
                    **{
                        metric: 0.78 - rank / 1000
                        for metric in ("roc", "prc", "accuracy", "f1", "mcc")
                    },
                }
            )
    pd.DataFrame(summary_rows).to_csv(benchmark / "cv_summary_metrics.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(benchmark / "matrix_test_metrics.csv", index=False)
    (benchmark / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "candidate_count": len(_candidates()),
                "suites": ["traditional", "chemprop"],
                "run_stage": "matrix_test",
                "test_snapshot_included": True,
                "candidate_matrix_frozen_before_test": True,
                "test_policy": "all frozen candidates receive shared-test metrics",
                "split_snapshot": {"files": {"manifest.json": f"sha-{protocol}"}},
            }
        ),
        encoding="utf-8",
    )


def _write_complete_results(output_root: Path) -> None:
    for protocol in ("scaffold", "random", "datasail"):
        _write_complete_protocol(output_root, protocol)


def _fake_figures(_frames: object, figures_dir: Path, **_kwargs: object) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("roc_valid", "roc_test"):
        path = figures_dir / f"{name}.png"
        path.write_bytes(b"figure")
        outputs[name] = path
    return outputs


def test_model_comparison_report_writes_compact_result_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_complete_results(tmp_path)
    destination = tmp_path / "report-output"
    monkeypatch.setattr(result_report, "write_selected_figures", _fake_figures)

    outputs = write_model_comparison_report(tmp_path, destination)

    all_candidates = pd.read_csv(outputs["all_candidates"])
    assert len(all_candidates) == 3 * len(_candidates())
    assert {"model_name", "model_type"}.issubset(all_candidates.columns)
    chemprop = all_candidates.loc[all_candidates["candidate_id"] == "chemprop_morgan"]
    assert set(chemprop["model_name"]) == {"Chemprop"}
    assert set(chemprop["model_type"]) == {"Graph + Morgan"}
    assert "top1" not in " ".join(all_candidates.columns).casefold()
    metrics_long = pd.read_csv(outputs["metrics_long"])
    gaps = pd.read_csv(outputs["validation_test_gap"])
    assert len(metrics_long) == 3 * len(_candidates()) * 2 * 5
    assert len(gaps) == 3 * len(_candidates()) * 5
    report = outputs["report_markdown"].read_text(encoding="utf-8")
    assert "No Top-1 model is selected" in report
    assert "Traditional model names" not in report
    assert str(tmp_path) not in report
    html = outputs["report_html"].read_text(encoding="utf-8")
    assert "figures/roc_valid.png" in html
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 58
    assert set(manifest["files"]) == {
        "all_candidates.csv",
        "figures/roc_test.png",
        "figures/roc_valid.png",
        "index.html",
        "metrics_long.csv",
        "report.md",
        "validation_test_gap.csv",
    }
    assert all(path.is_relative_to(destination) for path in outputs.values())


def test_model_comparison_report_rejects_cv_only_manifest(tmp_path: Path) -> None:
    _write_complete_results(tmp_path)
    path = tmp_path / "scaffold" / "benchmark" / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_stage"] = "cv_only"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen all-candidate"):
        write_model_comparison_report(tmp_path)
