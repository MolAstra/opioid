"""Estimator factory for the registered traditional benchmark candidates."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from opioid.benchmark.catalog import CandidateSpec

DEFAULT_SEED = 42


def build_estimator(candidate: CandidateSpec, *, seed: int = DEFAULT_SEED) -> Any:
    """Create a fresh estimator for one registered traditional candidate.

    Scaling is deliberately part of the estimator pipeline so every CV fold
    fits its own scaler using only its training rows. Tree ensembles and
    TabPFN receive raw feature matrices.
    """

    if not isinstance(candidate, CandidateSpec):
        raise TypeError("candidate must be a CandidateSpec")

    algorithm_id = candidate.algorithm_id
    parameters = candidate.algorithm.parameters
    if algorithm_id == "svm":
        estimator = SVC(
            C=float(parameters["C"]),
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    elif algorithm_id == "rf":
        estimator = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    elif algorithm_id == "xgb":
        estimator = _build_xgboost(parameters, seed)
    elif algorithm_id == "lgbm":
        estimator = _build_lightgbm(parameters, seed)
    elif algorithm_id == "tabpfn":
        estimator = _build_tabpfn(parameters, seed)
    elif algorithm_id.startswith("knn_k"):
        estimator = KNeighborsClassifier(
            n_neighbors=int(parameters["n_neighbors"]),
            weights=str(parameters["weights"]),
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
    elif algorithm_id.startswith("logreg_l"):
        estimator = _build_logistic_regression(parameters, seed)
    else:  # Defensive in case a malformed CandidateSpec was constructed manually.
        raise ValueError(f"Unsupported traditional algorithm {algorithm_id!r}")

    if candidate.algorithm.scale_features:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    return estimator


def _build_logistic_regression(parameters: Any, seed: int) -> LogisticRegression:
    penalty = str(parameters["penalty"])
    common = {
        "C": float(parameters["C"]),
        "class_weight": "balanced",
        "max_iter": int(parameters["max_iter"]),
        "random_state": seed,
    }
    if penalty == "l1":
        # scikit-learn 1.8 deprecates ``penalty``. An L1 ratio of one
        # preserves LASSO semantics while avoiding its deprecated interface.
        return LogisticRegression(solver="saga", l1_ratio=1.0, **common)
    if penalty == "l2":
        # The default L1 ratio of zero is standard L2 logistic regression.
        return LogisticRegression(solver="lbfgs", l1_ratio=0.0, **common)
    raise ValueError(f"Unsupported logistic-regression penalty {penalty!r}")


def _build_xgboost(parameters: Any, seed: int) -> Any:
    try:
        classifier = import_module("xgboost").XGBClassifier
    except Exception as error:  # pragma: no cover - installation dependent.
        raise RuntimeError("The xgb candidate requires xgboost to be installed") from error
    return classifier(
        n_estimators=int(parameters["n_estimators"]),
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )


def _build_lightgbm(parameters: Any, seed: int) -> Any:
    try:
        classifier = import_module("lightgbm").LGBMClassifier
    except Exception as error:  # pragma: no cover - installation dependent.
        raise RuntimeError("The lgbm candidate requires lightgbm to be installed") from error
    return classifier(
        n_estimators=int(parameters["n_estimators"]),
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=4,
    )


def _build_tabpfn(parameters: Any, seed: int) -> Any:
    try:
        classifier = import_module("tabpfn").TabPFNClassifier
    except Exception as error:  # pragma: no cover - installation dependent.
        raise RuntimeError("The tabpfn candidate requires tabpfn to be installed") from error
    return classifier(
        device="auto",
        random_state=seed,
        n_estimators=int(parameters["n_estimators"]),
        ignore_pretraining_limits=True,
    )


__all__ = ["DEFAULT_SEED", "build_estimator"]
