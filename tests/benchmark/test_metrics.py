from __future__ import annotations

import numpy as np
import pytest

from opioid.benchmark.metrics import compute_metrics


def test_compute_metrics_includes_matthews_correlation_coefficient() -> None:
    metrics = compute_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.8, 0.3, 0.9]), threshold=0.5
    )

    assert set(metrics) == {"roc", "prc", "accuracy", "f1", "mcc"}
    assert metrics["mcc"] == pytest.approx(0.0)
