from __future__ import annotations

from pathlib import Path

import pandas as pd

from opioid.screen.external_validation import (
    audit_training_overlap,
    load_external_source,
    summarize_accuracy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "workflow" / "70_external_validation" / "data" / "external_mor_20.csv"
TASK_ROOT = (
    REPO_ROOT
    / "workflow"
    / "10_build_data"
    / "data"
    / "dpbench"
    / "task_pool"
    / "muor_antagonism"
)


def test_fixed_external_source_and_overlap_contract() -> None:
    source = load_external_source(SOURCE)
    audited, input_files = audit_training_overlap(source, task_root=TASK_ROOT)

    assert len(audited) == 20
    assert audited["label"].value_counts().sort_index().to_dict() == {0: 10, 1: 10}
    overlaps = audited.loc[audited["training_overlap"]]
    assert set(overlaps["id"]) == {"EXT-N05", "EXT-N08", "EXT-N10"}
    assert overlaps["label"].eq(0).all()
    assert overlaps["training_label"].eq(0).all()
    assert len(input_files) == 12


def test_accuracy_summary_separates_all_and_nonoverlap_cohorts() -> None:
    scored = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "training_overlap": [False, False, False, True],
            "rf_pred_label": [1, 0, 1, 0],
            "tabpfn_pred_label": [1, 1, 0, 0],
            "lightgbm_pred_label": [1, 0, 0, 0],
            "strict_consensus_pred_label": [1, 0, 0, 0],
        }
    )

    summary = summarize_accuracy(scored)

    assert len(summary) == 8
    strict = summary.loc[summary["predictor"].eq("Strict three-model consensus")]
    full = strict.loc[strict["cohort"].eq("external_20_all")].iloc[0]
    nonoverlap = strict.loc[strict["cohort"].eq("external_17_nonoverlap")].iloc[0]
    assert (full["n"], full["correct"], full["accuracy"]) == (4, 3, 0.75)
    assert (full["sensitivity"], full["specificity"], full["balanced_accuracy"]) == (
        0.5,
        1.0,
        0.75,
    )
    assert (nonoverlap["n"], nonoverlap["correct"]) == (3, 2)
    assert nonoverlap["accuracy"] == 2 / 3
    assert nonoverlap["balanced_accuracy"] == 0.75
