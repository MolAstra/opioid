from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from opioid.benchmark.catalog import traditional_candidates
from opioid.benchmark.context import RunContext
from opioid.benchmark import runner


def _rows() -> pd.DataFrame:
    smiles = [
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCO",
        "CCCO",
        "CCCCO",
        "CCN",
        "CCCN",
        "CCCCN",
        "COC",
        "CCOC",
        "CC(=O)O",
        "CCC(=O)O",
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "Nc1ccccc1",
        "CCCl",
        "CCCCl",
    ]
    return pd.DataFrame({"smiles": smiles, "label": [index % 2 for index in range(len(smiles))]})


def _write_protocol(tmp_path: Path, *, include_test: bool) -> RunContext:
    split_dir = tmp_path / "splits" / "scaffold"
    split_dir.mkdir(parents=True)
    rows = _rows()
    for fold in range(5):
        valid_positions = list(range(fold * 4, fold * 4 + 4))
        valid = rows.iloc[valid_positions]
        train = rows.drop(index=valid_positions)
        fold_dir = split_dir / f"fold_{fold}"
        fold_dir.mkdir()
        train.to_csv(fold_dir / "train.csv", index=False)
        valid.to_csv(fold_dir / "valid.csv", index=False)
    if include_test:
        pd.DataFrame(
            {
                "smiles": ["C1CC1", "C1CCC1", "C1CCOC1", "C1CCNC1"],
                "label": [0, 1, 0, 1],
            }
        ).to_csv(split_dir / "test.csv", index=False)
    return RunContext(
        protocol="scaffold",
        split_dir=split_dir,
        output_dir=tmp_path / "outputs" / "scaffold" / "benchmark",
    )


def _single_candidate():
    return (traditional_candidates(feature_specs=("ecfp_1024",))[0],)


def _two_candidates():
    return traditional_candidates(feature_specs=("ecfp_1024",))[:2]


def test_cv_only_runner_never_loads_independent_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _write_protocol(tmp_path, include_test=True)
    monkeypatch.setattr(runner, "traditional_candidates", _single_candidate)
    monkeypatch.setattr(
        runner,
        "load_test_split",
        lambda _context: pytest.fail("CV-only benchmark must not load test.csv"),
    )
    monkeypatch.setattr(
        runner,
        "split_snapshot",
        lambda _context: pytest.fail("CV-only benchmark must not hash test.csv"),
    )

    result = runner.run_protocol_benchmark(
        context,
        suites=("traditional",),
        evaluate_test=False,
    )

    summary = pd.read_csv(result.summary_csv)
    assert set(summary["candidate_id"]) == {"svm__ecfp_1024"}
    assert "selected_top1" not in summary
    assert result.matrix_test_csv.is_file()
    assert pd.read_csv(result.matrix_test_csv).empty
    manifest = json.loads(result.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["test_snapshot_included"] is False
    assert "test.csv" not in manifest["split_snapshot"]["files"]
    assert manifest["test_policy"] == "CV-only run: independent test CSV was not loaded"


def test_frozen_candidate_matrix_has_five_shared_test_fold_scores_per_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _write_protocol(tmp_path, include_test=True)
    monkeypatch.setattr(runner, "traditional_candidates", _two_candidates)

    result = runner.run_protocol_benchmark(
        context,
        suites=("traditional",),
        evaluate_test=True,
    )

    matrix_test = pd.read_csv(result.matrix_test_csv)
    summary = pd.read_csv(result.summary_csv)
    candidate_ids = {candidate.candidate_id for candidate in _two_candidates()}
    assert len(matrix_test) == 10
    assert set(matrix_test["candidate_id"]) == candidate_ids
    for candidate_id in candidate_ids:
        candidate_test = matrix_test.loc[matrix_test["candidate_id"] == candidate_id]
        assert sorted(candidate_test["fold"].tolist()) == list(range(5))
        row = summary.loc[summary["candidate_id"] == candidate_id].iloc[0]
        assert row["test_roc_mean"] == pytest.approx(candidate_test["roc"].mean())
        assert row["test_roc_std"] == pytest.approx(candidate_test["roc"].std(ddof=1))
        assert row["test_mcc_mean"] == pytest.approx(candidate_test["mcc"].mean())
        assert row["test_n_folds"] == 5
    assert "selected_top1" not in summary
    assert "selection_rank" not in summary
    assert not list(context.artifacts_dir.glob("*.joblib"))
