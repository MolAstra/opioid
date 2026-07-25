from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflow"
    / "40_report_results"
    / "plot_family_bars.py"
)
SPEC = importlib.util.spec_from_file_location("workflow40_family_bars", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
family_bars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(family_bars)


def _candidate_rows() -> pd.DataFrame:
    candidates = [
        ("svm__a", "svm", "svm", "ecfp", "SVM (RBF)", "ECFP"),
        ("rf__a", "rf", "rf", "ecfp", "Random Forest", "ECFP"),
        ("rf__b", "rf", "rf", "descriptor", "Random Forest", "Descriptor"),
        ("xgb__a", "xgb", "xgb", "ecfp", "XGBoost", "ECFP"),
        ("xgb__b", "xgb", "xgb", "descriptor", "XGBoost", "Descriptor"),
        ("lgbm__a", "lgbm", "lgbm", "ecfp", "LightGBM", "ECFP"),
        ("tabpfn__a", "tabpfn", "tabpfn", "ecfp", "TabPFN", "ECFP"),
        ("knn_k3__a", "knn_k3", "knn_k3", "ecfp", "KNN k=3", "ECFP"),
        ("knn_k5__a", "knn_k5", "knn_k5", "descriptor", "KNN k=5", "Descriptor"),
        (
            "logreg_l1_c0_1__a",
            "logreg_l1_c0_1",
            "logreg_l1_c0_1",
            "ecfp",
            "Logistic L1",
            "ECFP",
        ),
        (
            "logreg_l2_c0_1__a",
            "logreg_l2_c0_1",
            "logreg_l2_c0_1",
            "descriptor",
            "Logistic L2",
            "Descriptor",
        ),
        ("chemprop_default", "chemprop", "default", "graph", "Chemprop", "Graph"),
        (
            "chemprop_morgan",
            "chemprop",
            "morgan",
            "graph_morgan",
            "Chemprop",
            "Graph + Morgan",
        ),
    ]
    rows = []
    for protocol in family_bars.PROTOCOLS:
        for index, (
            candidate_id,
            algorithm,
            preset,
            representation,
            model_name,
            model_type,
        ) in enumerate(candidates):
            row = {
                "protocol": protocol,
                "family": "chemprop" if algorithm == "chemprop" else "traditional",
                "algorithm": algorithm,
                "preset": preset,
                "representation": representation,
                "candidate_id": candidate_id,
                "model_name": model_name,
                "model_type": model_type,
                "n_folds": 5,
                "test_n_folds": 5,
            }
            for metric_index, metric in enumerate(family_bars.METRICS):
                row[f"valid_{metric}_mean"] = 0.30 + 0.01 * index + 0.005 * metric_index
                row[f"valid_{metric}_std"] = 0.02
                row[f"test_{metric}_mean"] = 0.40 + 0.01 * index + 0.005 * metric_index
                row[f"test_{metric}_std"] = 0.03
            rows.append(row)
    frame = pd.DataFrame(rows)

    def assign(candidate_id: str, protocol: str, column: str, value: float) -> None:
        mask = (frame["candidate_id"] == candidate_id) & (frame["protocol"] == protocol)
        frame.loc[mask, column] = value

    assign("rf__a", "scaffold", "test_roc_mean", 0.95)
    assign("rf__b", "scaffold", "test_roc_mean", 0.90)
    assign("rf__a", "scaffold", "valid_roc_mean", 0.923)
    assign("rf__b", "scaffold", "valid_roc_mean", 0.223)
    assign("rf__a", "scaffold", "test_prc_mean", 0.20)
    assign("rf__b", "scaffold", "test_prc_mean", 0.97)
    assign("rf__a", "random", "test_roc_mean", 0.10)
    assign("rf__b", "random", "test_roc_mean", 0.98)
    assign("knn_k3__a", "datasail", "test_f1_mean", 0.30)
    assign("knn_k5__a", "datasail", "test_f1_mean", 0.93)
    assign("logreg_l1_c0_1__a", "datasail", "test_accuracy_mean", 0.20)
    assign("logreg_l2_c0_1__a", "datasail", "test_accuracy_mean", 0.91)
    assign("xgb__a", "scaffold", "test_accuracy_mean", 0.88)
    assign("xgb__b", "scaffold", "test_accuracy_mean", 0.88)
    return frame


def _selection(
    selected: pd.DataFrame,
    metric: str,
    protocol: str,
    model_family: str,
) -> pd.Series:
    rows = selected.loc[
        (selected["metric_key"] == metric)
        & (selected["protocol"] == protocol)
        & (selected["model_family"] == model_family)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_metric_protocol_and_family_selection_is_explicit() -> None:
    frame = _candidate_rows()
    frame["model_family"] = frame["algorithm"].map(family_bars.model_family)
    selected = family_bars.select_family_candidates(frame)

    assert len(selected) == 4 * 3 * 8
    scaffold_rf_auroc = _selection(selected, "roc", "scaffold", "RF")
    assert scaffold_rf_auroc["selected_candidate_id"] == "rf__a"
    assert scaffold_rf_auroc["valid_mean"] == pytest.approx(0.923)
    assert _selection(selected, "prc", "scaffold", "RF")[
        "selected_candidate_id"
    ] == "rf__a"
    assert _selection(selected, "roc", "random", "RF")[
        "selected_candidate_id"
    ] == "rf__b"
    assert _selection(selected, "f1", "datasail", "KNN")[
        "selected_candidate_id"
    ] == "knn_k5__a"
    assert _selection(selected, "accuracy", "datasail", "LR")[
        "selected_candidate_id"
    ] == "logreg_l2_c0_1__a"
    assert _selection(selected, "accuracy", "scaffold", "XGB")[
        "selected_candidate_id"
    ] == "xgb__b"


def test_package_writes_four_figures_and_auditable_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "all_candidates.csv"
    _candidate_rows().to_csv(input_path, index=False)
    output_dir = tmp_path / "model_family_bars"

    outputs = family_bars.write_package(input_path, output_dir)

    assert set(outputs) == {
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "selected_candidates",
        "manifest",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    selected = pd.read_csv(outputs["selected_candidates"])
    assert len(selected) == 96
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["selection_scope"] == "validation_selected_descriptive_family_view"
    assert manifest["independent_evaluation"] is False
    assert manifest["deployment_artifact"] is False
    assert manifest["bars_per_figure"] == 48
    assert manifest["selection_rows"] == 96
    assert manifest["presentation"] == {
        "dpi": 600,
        "figure_size_inches": [15.5, 7.2],
        "font_family": "Arial",
        "font_weight": "normal",
        "legend_columns": 3,
        "model_spacing": 0.93,
        "palette": {
            "datasail_test": "#168B54",
            "datasail_valid": "#7DDB8A",
            "random_test": "#F05A00",
            "random_valid": "#FFB347",
            "scaffold_test": "#0077B6",
            "scaffold_valid": "#62C7F2",
        },
        "bar_width": 0.135,
        "value_label_decimals": 2,
    }
    assert set(manifest["files"]) == {
        "accuracy.png",
        "auprc.png",
        "auroc.png",
        "f1.png",
        "selected_candidates.csv",
    }
    with pytest.raises(FileExistsError, match="--replace"):
        family_bars.write_package(input_path, output_dir)


def test_unknown_algorithm_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unregistered algorithm"):
        family_bars.model_family("mystery")
