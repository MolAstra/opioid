from __future__ import annotations

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.estimators import build_estimator


def _candidate(candidate_id: str):
    return next(item for item in traditional_candidates() if item.candidate_id == candidate_id)


def test_logistic_presets_use_explicit_l1_and_l2_regularization() -> None:
    l1 = build_estimator(_candidate("logreg_l1_c1__ecfp_1024"), seed=42).named_steps["model"]
    l2 = build_estimator(_candidate("logreg_l2_c1__ecfp_1024"), seed=42).named_steps["model"]

    assert l1.solver == "saga"
    assert l1.l1_ratio == 1.0
    assert l1.max_iter == 10_000
    assert l2.solver == "lbfgs"
    assert l2.l1_ratio == 0.0
    assert l2.max_iter == 10_000
