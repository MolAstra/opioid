from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from opioid.evaluation.similarity import (
    assert_scaffold_test_isolation,
    diagnose_cv_folds,
    diagnose_protocol_split,
    write_cv_fold_similarity_diagnostics,
    write_similarity_diagnostics,
)


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
    return pd.DataFrame(
        {
            "sample_id": [f"row-{index}" for index in range(len(smiles))],
            "smiles": smiles,
            "scaffold": ["c1ccccc1" if "c1ccccc1" in value else "" for value in smiles],
            "label": [index % 2 for index in range(len(smiles))],
        }
    )


def _write_dpbench_protocol(root: Path, protocol: str = "datasail") -> Path:
    split_dir = root / f"processed_{protocol}"
    split_dir.mkdir(parents=True)
    rows = _rows()
    for fold in range(5):
        valid_positions = list(range(fold * 4, fold * 4 + 4))
        rows.iloc[valid_positions].to_csv(split_dir / f"valid_fold-{fold}.csv", index=False)
        rows.drop(index=valid_positions).to_csv(split_dir / f"train_fold-{fold}.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": ["test-0", "test-1", "test-2", "test-3"],
            "smiles": ["C1CC1", "C1CCC1", "C1CCOC1", "C1CCNC1"],
            "scaffold": ["C1CC1", "C1CCC1", "C1CCOC1", "C1CCNC1"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(split_dir / "test.csv", index=False)
    (split_dir / "diagnostics.json").write_text('{"test_rows": 4}\n')
    return split_dir


def test_dpbench_similarity_diagnostics_cover_test_and_all_folds(tmp_path: Path) -> None:
    split_dir = _write_dpbench_protocol(tmp_path)
    result = diagnose_protocol_split(split_dir, protocol="datasail")

    assert len(result.folds) == 5
    assert len(result.test.per_target) == 4
    assert result.datasail_diagnostics == {"test_rows": 4}
    assert result.fold_summary["n_cross_pairs"].gt(0).all()
    outputs = write_similarity_diagnostics(result, tmp_path / "diagnostics")
    assert all(path.is_file() for path in outputs.values())
    summary = (tmp_path / "diagnostics" / "similarity_summary.json").read_text(encoding="utf-8")
    assert "descriptive post-hoc diagnostics" in summary


def test_scaffold_protocol_rejects_development_test_scaffold_overlap(tmp_path: Path) -> None:
    split_dir = _write_dpbench_protocol(tmp_path, protocol="scaffold")
    test = pd.read_csv(split_dir / "test.csv")
    test.loc[0, "smiles"] = "c1ccccc1"
    test.loc[0, "scaffold"] = "c1ccccc1"
    test.to_csv(split_dir / "test.csv", index=False)

    with pytest.raises(AssertionError, match="Scaffold development/test isolation failed"):
        diagnose_protocol_split(split_dir, protocol="scaffold")

    unchecked = diagnose_protocol_split(split_dir, protocol="datasail")
    with pytest.raises(AssertionError, match="Scaffold development/test isolation failed"):
        assert_scaffold_test_isolation(unchecked)


def test_supplemental_fold_diagnostics_do_not_read_test(tmp_path: Path) -> None:
    split_dir = _write_dpbench_protocol(tmp_path)
    (split_dir / "test.csv").unlink()

    result = diagnose_cv_folds(split_dir, protocol="datasail")
    outputs = write_cv_fold_similarity_diagnostics(result, tmp_path / "fold-diagnostics")

    assert len(result.folds) == 5
    assert all(path.is_file() for path in outputs.values())
    assert not any("test" in path.name for path in outputs.values())
    summary = (tmp_path / "fold-diagnostics" / "similarity_summary.json").read_text(
        encoding="utf-8"
    )
    assert '"scope": "train_validation_folds_only"' in summary
