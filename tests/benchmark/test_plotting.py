from __future__ import annotations

from pathlib import Path

import pandas as pd

from opioid.benchmark.plotting import (
    load_protocol_frames,
    load_summary,
    validation_candidate_order,
    write_selected_figures,
)


def _metric_values(valid: float, test: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric, offset in (
        ("roc", 0.0),
        ("prc", -0.03),
        ("accuracy", -0.07),
        ("f1", -0.09),
        ("mcc", -0.20),
    ):
        values[f"valid_{metric}_mean"] = valid + offset
        values[f"valid_{metric}_std"] = 0.02
        values[f"test_{metric}_mean"] = test + offset
        values[f"test_{metric}_std"] = 0.03
    return values


def _summary_rows(offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "status": "complete",
            "family": "traditional",
            "algorithm": "rf",
            "preset": "default",
            "representation": "ecfp_1024",
            "candidate_id": "rf__ecfp_1024",
            **_metric_values(0.84 + offset, 0.82 + offset),
        },
        {
            "status": "complete",
            "family": "traditional",
            "algorithm": "rf",
            "preset": "default",
            "representation": "ecfp_2048",
            "candidate_id": "rf__ecfp_2048",
            **_metric_values(0.81 + offset, 0.79 + offset),
        },
        {
            "status": "complete",
            "family": "chemprop",
            "algorithm": "chemprop",
            "preset": "morgan",
            "representation": "morgan_binary",
            "candidate_id": "chemprop_morgan",
            **_metric_values(0.79 + offset, 0.77 + offset),
        },
        {
            "status": "pending",
            "family": "external",
            "algorithm": "not_complete",
            "preset": "pending",
            "representation": "external",
            "candidate_id": "not_complete",
            **_metric_values(float("nan"), float("nan")),
        },
    ]


def _write_summary(output_root: Path, protocol: str = "scaffold", offset: float = 0.0) -> Path:
    benchmark = output_root / protocol / "benchmark"
    benchmark.mkdir(parents=True)
    summary_csv = benchmark / "cv_summary_metrics.csv"
    pd.DataFrame(_summary_rows(offset)).to_csv(summary_csv, index=False)
    return summary_csv


def _write_all_summaries(output_root: Path) -> None:
    for index, protocol in enumerate(("scaffold", "random", "datasail")):
        _write_summary(output_root, protocol, offset=index / 100)


def test_load_summary_excludes_incomplete_rows(tmp_path: Path) -> None:
    result = load_summary(_write_summary(tmp_path))

    assert set(result["candidate_id"]) == {
        "rf__ecfp_1024",
        "rf__ecfp_2048",
        "chemprop_morgan",
    }
    assert {"model_name", "model_type"}.issubset(result.columns)
    assert "selected_top1" not in result


def test_validation_order_uses_mean_validation_auroc_only(tmp_path: Path) -> None:
    _write_all_summaries(tmp_path)
    frames = load_protocol_frames(tmp_path)

    order = validation_candidate_order(frames)

    assert order == ["rf__ecfp_1024", "rf__ecfp_2048", "chemprop_morgan"]


def test_complete_figure_set_is_flat_and_unified(tmp_path: Path) -> None:
    _write_all_summaries(tmp_path)
    destination = tmp_path / "report-output" / "figures"

    outputs = write_selected_figures(load_protocol_frames(tmp_path), destination)

    expected = {
        *(
            f"{metric}_{stage}"
            for stage in ("valid", "test")
            for metric in ("roc", "prc", "accuracy", "f1", "mcc")
        ),
        *(f"{metric}_validation_vs_test" for metric in ("roc", "prc", "accuracy", "f1", "mcc")),
        *(
            f"{protocol}_{stage}_overview"
            for protocol in ("scaffold", "random", "datasail")
            for stage in ("valid", "test")
        ),
    }
    assert set(outputs) == expected
    assert len(outputs) == 21
    assert all(path.parent == destination for path in outputs.values())
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())


def test_single_protocol_validation_plot_writes_one_overview(tmp_path: Path) -> None:
    _write_summary(tmp_path)
    destination = tmp_path / "figures"

    outputs = write_selected_figures(
        load_protocol_frames(tmp_path, ("scaffold",)),
        destination,
        stages=("valid",),
    )

    assert set(outputs) == {"scaffold_valid_overview"}
