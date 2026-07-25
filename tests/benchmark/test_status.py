from __future__ import annotations

from opioid.benchmark.status import StatusStore


def test_status_resume_requires_matching_input_fingerprint(tmp_path) -> None:
    store = StatusStore(tmp_path / "status.json")
    store.complete("cv.scaffold.rf__ecfp_2048.fold_0", {"valid_roc": 0.8}, input_fingerprint="a")

    assert store.completed("cv.scaffold.rf__ecfp_2048.fold_0", input_fingerprint="a") is not None
    assert store.completed("cv.scaffold.rf__ecfp_2048.fold_0", input_fingerprint="changed") is None
