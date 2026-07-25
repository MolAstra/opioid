"""Leakage-safe reviewer benchmark APIs."""

from opioid.benchmark.catalog import (
    ALGORITHM_SPECS,
    CandidateSpec,
    AlgorithmSpec,
    traditional_candidates,
)
from opioid.benchmark.estimators import build_estimator
from opioid.benchmark.context import EVALUATION_PROTOCOLS, RunContext
from opioid.benchmark.runner import (
    BenchmarkRunResult,
    run_all_protocol_benchmarks,
    run_protocol_benchmark,
)
from opioid.benchmark.family_selection import (
    MODEL_FAMILIES,
    model_family,
    rank_test_informed_families,
    select_validation_representatives,
)

__all__ = [
    "ALGORITHM_SPECS",
    "AlgorithmSpec",
    "BenchmarkRunResult",
    "CandidateSpec",
    "EVALUATION_PROTOCOLS",
    "MODEL_FAMILIES",
    "RunContext",
    "build_estimator",
    "model_family",
    "rank_test_informed_families",
    "run_all_protocol_benchmarks",
    "run_protocol_benchmark",
    "select_validation_representatives",
    "traditional_candidates",
]
