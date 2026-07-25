from __future__ import annotations

import pandas as pd
import pytest

from opioid.benchmark.display_names import (
    MODEL_DISPLAY_CATALOG,
    add_model_display_columns,
    get_model_display,
    model_catalog_frame,
)


def test_complete_catalog_has_unique_two_part_names() -> None:
    catalog = model_catalog_frame()

    assert len(catalog) == 58
    assert catalog["candidate_id"].is_unique
    assert not catalog.duplicated(["model_name", "model_type"]).any()
    assert catalog["family"].value_counts().to_dict() == {"traditional": 55, "chemprop": 3}


def test_publication_names_cover_traditional_and_chemprop_candidates() -> None:
    assert get_model_display("rf__ecfp_1024").plot_label == ("Random Forest\nECFP4 (1024-bit)")
    assert get_model_display("logreg_l1_c0_1__rdkit2d_normalized_200").plot_label == (
        "Logistic Regression (L1, C=0.1)\nRDKit2D (200-d)"
    )
    assert get_model_display("chemprop_default").plot_label == "Chemprop\nGraph"
    assert get_model_display("chemprop_morgan").plot_label == "Chemprop\nGraph + Morgan"
    assert get_model_display("chemprop_morgan_rdkit2d").plot_label == (
        "Chemprop\nGraph + Morgan + RDKit2D"
    )


def test_display_columns_preserve_candidate_id_and_reject_unknown_candidates() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["rf__ecfp_1024", "chemprop_default"], "score": [0.8, 0.7]}
    )

    result = add_model_display_columns(frame)

    assert result.columns.tolist() == ["candidate_id", "model_name", "model_type", "score"]
    assert result["candidate_id"].tolist() == frame["candidate_id"].tolist()
    with pytest.raises(ValueError, match="without display names"):
        add_model_display_columns(pd.DataFrame({"candidate_id": ["unknown_complete"]}))
    assert len(MODEL_DISPLAY_CATALOG) == 58
