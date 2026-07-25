"""Publication-facing names for the frozen reviewer candidate catalog."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.chemprop import chemprop_candidates


@dataclass(frozen=True)
class ModelDisplay:
    """A stable two-part display identity for one benchmark candidate."""

    model_name: str
    model_type: str
    family: str

    @property
    def plot_label(self) -> str:
        return f"{self.model_name}\n{self.model_type}"


_MODEL_NAMES = {
    "svm": "SVM (RBF)",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "tabpfn": "TabPFN",
    "knn_k3": "k-Nearest Neighbors (k=3)",
    "knn_k5": "k-Nearest Neighbors (k=5)",
    "logreg_l1_c0_1": "Logistic Regression (L1, C=0.1)",
    "logreg_l1_c1": "Logistic Regression (L1, C=1)",
    "logreg_l2_c0_1": "Logistic Regression (L2, C=0.1)",
    "logreg_l2_c1": "Logistic Regression (L2, C=1)",
}

_MODEL_TYPES = {
    "ecfp_1024": "ECFP4 (1024-bit)",
    "ecfp_2048": "ECFP4 (2048-bit)",
    "rdkit2d_normalized_200": "RDKit2D (200-d)",
    "ecfp_1024_rdkit2d_normalized_200": "ECFP4 (1024-bit) + RDKit2D (200-d)",
    "ecfp_2048_rdkit2d_normalized_200": "ECFP4 (2048-bit) + RDKit2D (200-d)",
}

_CHEMPROP_TYPES = {
    "chemprop_default": "Graph",
    "chemprop_morgan": "Graph + Morgan",
    "chemprop_morgan_rdkit2d": "Graph + Morgan + RDKit2D",
}


def _build_catalog() -> dict[str, ModelDisplay]:
    catalog = {
        candidate.candidate_id: ModelDisplay(
            model_name=_MODEL_NAMES[candidate.algorithm_id],
            model_type=_MODEL_TYPES[candidate.representation_id],
            family=candidate.family,
        )
        for candidate in traditional_candidates()
    }
    catalog.update(
        {
            candidate.candidate_id: ModelDisplay(
                model_name="Chemprop",
                model_type=_CHEMPROP_TYPES[candidate.candidate_id],
                family=candidate.family,
            )
            for candidate in chemprop_candidates()
        }
    )
    pairs = {(item.model_name, item.model_type) for item in catalog.values()}
    if len(pairs) != len(catalog):
        raise RuntimeError("Model display names must uniquely identify every candidate")
    return catalog


MODEL_DISPLAY_CATALOG = _build_catalog()


def get_model_display(candidate_id: str) -> ModelDisplay:
    """Resolve one registered candidate's publication-facing identity."""

    try:
        return MODEL_DISPLAY_CATALOG[str(candidate_id)]
    except KeyError as error:
        raise ValueError(f"No display name is registered for candidate {candidate_id!r}") from error


def add_model_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized ``model_name`` and ``model_type`` columns."""

    if "candidate_id" not in frame:
        raise ValueError("Model display names require a candidate_id column")
    work = frame.copy()
    candidate_ids = work["candidate_id"].astype(str)
    unknown = sorted(set(candidate_ids).difference(MODEL_DISPLAY_CATALOG))
    if unknown:
        raise ValueError(f"Completed results contain candidates without display names: {unknown}")
    displays = candidate_ids.map(MODEL_DISPLAY_CATALOG)
    for column in ("model_name", "model_type"):
        if column in work:
            work = work.drop(columns=column)
    insert_at = work.columns.get_loc("candidate_id") + 1
    work.insert(insert_at, "model_name", displays.map(lambda item: item.model_name))
    work.insert(insert_at + 1, "model_type", displays.map(lambda item: item.model_type))
    return work


def model_catalog_frame() -> pd.DataFrame:
    """Return the complete display catalog in stable candidate registration order."""

    return pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "model_name": display.model_name,
                "model_type": display.model_type,
                "family": display.family,
            }
            for candidate_id, display in MODEL_DISPLAY_CATALOG.items()
        ]
    )


__all__ = [
    "MODEL_DISPLAY_CATALOG",
    "ModelDisplay",
    "add_model_display_columns",
    "get_model_display",
    "model_catalog_frame",
]
