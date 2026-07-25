"""Auditable model-family selection views derived from the frozen benchmark matrix."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


MODEL_FAMILIES: Final = (
    "SVM",
    "RF",
    "XGB",
    "LightGBM",
    "TabPFN",
    "Chemprop",
    "KNN",
    "LR",
)


def model_family(algorithm: str) -> str:
    """Map a registered algorithm or preset identifier to its display family."""

    if algorithm.startswith("knn_"):
        return "KNN"
    if algorithm.startswith("logreg_"):
        return "LR"
    mapping = {
        "svm": "SVM",
        "rf": "RF",
        "xgb": "XGB",
        "lgbm": "LightGBM",
        "tabpfn": "TabPFN",
        "chemprop": "Chemprop",
    }
    try:
        return mapping[algorithm]
    except KeyError as error:
        raise ValueError(
            f"Unregistered algorithm cannot be assigned to a family: {algorithm}"
        ) from error


def select_validation_representatives(
    frame: pd.DataFrame,
    *,
    protocol: str,
) -> pd.DataFrame:
    """Select one candidate per family using mean validation AUROC only."""

    required = {
        "protocol",
        "algorithm",
        "candidate_id",
        "valid_roc_mean",
        "test_roc_mean",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Candidate table is missing selection columns: {missing}")
    if frame.duplicated(["protocol", "candidate_id"]).any():
        raise ValueError("Candidate table contains duplicate protocol/candidate rows")

    work = frame.loc[frame["protocol"].astype(str) == protocol].copy()
    if work.empty:
        raise ValueError(f"Candidate table does not contain protocol={protocol!r}")
    work["model_family"] = work["algorithm"].astype(str).map(model_family)
    if set(work["model_family"]) != set(MODEL_FAMILIES):
        missing_families = sorted(set(MODEL_FAMILIES).difference(work["model_family"]))
        raise ValueError(f"Candidate table is missing model families: {missing_families}")
    numeric = work[["valid_roc_mean", "test_roc_mean"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Candidate selection metrics must be finite")
    work[["valid_roc_mean", "test_roc_mean"]] = numeric

    selected_rows: list[pd.Series] = []
    for family in MODEL_FAMILIES:
        candidates = work.loc[work["model_family"] == family]
        best_value = float(candidates["valid_roc_mean"].max())
        selected_rows.append(
            candidates.loc[candidates["valid_roc_mean"].eq(best_value)]
            .sort_values("candidate_id", kind="stable")
            .iloc[0]
        )
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["validation_selection_metric"] = "valid_roc_mean"
    return selected


def rank_test_informed_families(
    representatives: pd.DataFrame,
    *,
    top_n: int = 3,
) -> pd.DataFrame:
    """Rank validation-selected family representatives by shared-test AUROC."""

    if top_n <= 0 or top_n > len(MODEL_FAMILIES):
        raise ValueError(f"top_n must be between 1 and {len(MODEL_FAMILIES)}")
    if set(representatives["model_family"].astype(str)) != set(MODEL_FAMILIES):
        raise ValueError("Test ranking requires exactly one representative for every model family")
    if representatives["model_family"].duplicated().any():
        raise ValueError("Test ranking received duplicate model families")

    ranked = representatives.sort_values(
        ["test_roc_mean", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "test_roc_rank", np.arange(1, len(ranked) + 1, dtype=int))
    ranked["selected_for_screening"] = ranked["test_roc_rank"].le(top_n)
    ranked["test_ranking_metric"] = "test_roc_mean"
    return ranked


__all__ = [
    "MODEL_FAMILIES",
    "model_family",
    "rank_test_informed_families",
    "select_validation_representatives",
]
