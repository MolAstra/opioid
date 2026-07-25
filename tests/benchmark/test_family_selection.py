from __future__ import annotations

import pandas as pd

from opioid.benchmark.family_selection import (
    MODEL_FAMILIES,
    rank_test_informed_families,
    select_validation_representatives,
)


def test_datasail_two_stage_selection_includes_chemprop() -> None:
    algorithms = {
        "SVM": "svm",
        "RF": "rf",
        "XGB": "xgb",
        "LightGBM": "lgbm",
        "TabPFN": "tabpfn",
        "Chemprop": "chemprop",
        "KNN": "knn_k3",
        "LR": "logreg_l1_c0_1",
    }
    test_scores = {
        "SVM": 0.48,
        "RF": 0.70,
        "XGB": 0.69,
        "LightGBM": 0.72,
        "TabPFN": 0.71,
        "Chemprop": 0.66,
        "KNN": 0.49,
        "LR": 0.45,
    }
    rows = []
    for index, family in enumerate(MODEL_FAMILIES):
        algorithm = algorithms[family]
        rows.extend(
            [
                {
                    "protocol": "datasail",
                    "algorithm": algorithm,
                    "candidate_id": f"{algorithm}__low",
                    "valid_roc_mean": 0.5,
                    "test_roc_mean": 0.99,
                },
                {
                    "protocol": "datasail",
                    "algorithm": algorithm,
                    "candidate_id": f"{algorithm}__selected",
                    "valid_roc_mean": 0.8 + index / 100,
                    "test_roc_mean": test_scores[family],
                },
            ]
        )

    representatives = select_validation_representatives(
        pd.DataFrame(rows), protocol="datasail"
    )
    ranked = rank_test_informed_families(representatives)

    assert set(representatives["model_family"]) == set(MODEL_FAMILIES)
    assert (
        representatives.set_index("model_family").loc["Chemprop", "candidate_id"]
        == "chemprop__selected"
    )
    assert ranked.loc[ranked["selected_for_screening"], "model_family"].tolist() == [
        "LightGBM",
        "TabPFN",
        "RF",
    ]
    assert not ranked.loc[
        ranked["model_family"] == "Chemprop", "selected_for_screening"
    ].item()
