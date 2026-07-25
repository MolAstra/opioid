"""Stable catalog of the traditional ML × molecular-representation matrix."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from opioid.chemistry.representations import (
    FEATURE_SPECS,
    FeatureSpec,
    get_feature_spec,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One algorithm/preset that can be paired with every feature representation."""

    id: str
    label: str
    scale_features: bool
    parameters: Mapping[str, Any]

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "scale_features": self.scale_features,
            "parameters": dict(self.parameters),
        }


def _algorithm(
    algorithm_id: str,
    label: str,
    *,
    scale_features: bool,
    **parameters: Any,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        id=algorithm_id,
        label=label,
        scale_features=scale_features,
        # Keep this a normal dict so dataclasses.asdict() can serialize a
        # CandidateSpec into the runtime manifest. The catalog itself is not
        # exported mutably, and estimator construction never mutates it.
        parameters=dict(parameters),
    )


_ALGORITHM_SPECS = {
    # Existing conventional benchmark algorithms.
    "svm": _algorithm("svm", "SVM (RBF)", scale_features=True, C=1.0),
    "rf": _algorithm("rf", "Random forest", scale_features=False, n_estimators=600),
    "xgb": _algorithm("xgb", "XGBoost", scale_features=False, n_estimators=600),
    "lgbm": _algorithm("lgbm", "LightGBM", scale_features=False, n_estimators=600),
    "tabpfn": _algorithm("tabpfn", "TabPFN", scale_features=False, n_estimators=8),
    # Requested k-nearest-neighbour presets.
    "knn_k3": _algorithm(
        "knn_k3",
        "k-nearest neighbors (k=3)",
        scale_features=True,
        n_neighbors=3,
        weights="distance",
    ),
    "knn_k5": _algorithm(
        "knn_k5",
        "k-nearest neighbors (k=5)",
        scale_features=True,
        n_neighbors=5,
        weights="distance",
    ),
    # LASSO/L1 logistic regression presets.
    "logreg_l1_c0_1": _algorithm(
        "logreg_l1_c0_1",
        "L1 logistic regression (C=0.1)",
        scale_features=True,
        penalty="l1",
        C=0.1,
        max_iter=10_000,
    ),
    "logreg_l1_c1": _algorithm(
        "logreg_l1_c1",
        "L1 logistic regression (C=1)",
        scale_features=True,
        penalty="l1",
        C=1.0,
        max_iter=10_000,
    ),
    # Standard L2 logistic regression presets.
    "logreg_l2_c0_1": _algorithm(
        "logreg_l2_c0_1",
        "L2 logistic regression (C=0.1)",
        scale_features=True,
        penalty="l2",
        C=0.1,
        max_iter=10_000,
    ),
    "logreg_l2_c1": _algorithm(
        "logreg_l2_c1",
        "L2 logistic regression (C=1)",
        scale_features=True,
        penalty="l2",
        C=1.0,
        max_iter=10_000,
    ),
}

# Preserve the registration order to make candidate IDs and result tables stable.
ALGORITHM_SPECS: Mapping[str, AlgorithmSpec] = MappingProxyType(_ALGORITHM_SPECS)


@dataclass(frozen=True)
class CandidateSpec:
    """One independently trainable traditional algorithm/representation pairing."""

    candidate_id: str
    algorithm: AlgorithmSpec
    feature_spec: FeatureSpec
    family: str = "traditional"

    @property
    def id(self) -> str:
        """Alias for callers that use generic candidate IDs."""

        return self.candidate_id

    @property
    def algorithm_id(self) -> str:
        return self.algorithm.id

    @property
    def representation_id(self) -> str:
        return self.feature_spec.id

    def manifest(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "algorithm": self.algorithm.manifest(),
            "representation": self.feature_spec.manifest(),
        }


def traditional_candidates(
    feature_specs: Iterable[FeatureSpec | str] | None = None,
) -> tuple[CandidateSpec, ...]:
    """Return the complete 11 algorithm × representation candidate matrix.

    With the five registered representation specs, this always returns 55
    candidates. A caller can pass a feature subset for focused smoke tests or
    a reduced benchmark run while retaining stable candidate identifiers.
    """

    selected_features = _resolve_feature_specs(feature_specs)
    candidates = tuple(
        CandidateSpec(
            candidate_id=f"{algorithm.id}__{feature_spec.id}",
            algorithm=algorithm,
            feature_spec=feature_spec,
        )
        for algorithm in ALGORITHM_SPECS.values()
        for feature_spec in selected_features
    )
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise RuntimeError("Traditional candidate catalog contains duplicate candidate IDs")
    return candidates


def _resolve_feature_specs(
    feature_specs: Iterable[FeatureSpec | str] | None,
) -> tuple[FeatureSpec, ...]:
    if feature_specs is None:
        return tuple(FEATURE_SPECS.values())
    selected = tuple(get_feature_spec(spec) for spec in feature_specs)
    if not selected:
        raise ValueError("At least one feature spec is required")
    if len({spec.id for spec in selected}) != len(selected):
        raise ValueError("Feature specs must not contain duplicate identifiers")
    return selected


__all__ = [
    "ALGORITHM_SPECS",
    "AlgorithmSpec",
    "CandidateSpec",
    "traditional_candidates",
]
