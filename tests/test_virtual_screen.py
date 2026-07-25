from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opioid.screen import virtual_screen


def _write_sources(root: Path) -> None:
    root.mkdir()
    pd.DataFrame(
        {
            "smiles": ["CCO", "OCC", "not-smiles", "CC", "C[C@H](O)F"],
            "gpcrdb_cid": [1, 2, 3, 4, 5],
        }
    ).to_csv(root / "gpcrdb.csv", index=False)
    pd.DataFrame({"zinc_id": ["a", "b"], "smiles": ["C[C@H](O)F", "CCC"]}).to_csv(
        root / "zinc_scaffold.csv", index=False
    )
    pd.DataFrame({"SMILES": ["CCCC"], "Score": [0.8]}).to_csv(
        root / "reinvent_stage2_1.csv", index=False
    )
    pd.DataFrame({"smiles": ["CCCCC"], "SAScore": [0.7]}).to_csv(
        root / "opioid_ouroboros.csv", index=False
    )


def test_clean_sources_uses_canonical_overlap_and_deduplication(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    _write_sources(source_root)

    global_candidates, audit, representatives = virtual_screen.clean_screening_sources(
        source_root,
        training_smiles=["CC"],
    )

    gpcrdb_status = audit.loc[audit["source"] == "gpcrdb", "status"].value_counts().to_dict()
    assert gpcrdb_status == {
        "eligible": 2,
        "within_source_duplicate": 1,
        "invalid_smiles": 1,
        "training_overlap": 1,
    }
    assert len(representatives["gpcrdb"]) == 2
    shared = global_candidates.loc[global_candidates["source_count"] == 2].iloc[0]
    assert shared["sources"] == "gpcrdb;zinc"
    assert len(global_candidates) == 5


def test_consensus_is_fixed_all_three_threshold() -> None:
    frame = pd.DataFrame(
        {
            virtual_screen.SCORE_COLUMNS[virtual_screen.RF_ID]: [0.5, 0.9, 0.1],
            virtual_screen.SCORE_COLUMNS[virtual_screen.TABPFN_ID]: [0.5, 0.4, 0.2],
            virtual_screen.SCORE_COLUMNS[virtual_screen.LGBM_ID]: [0.5, 0.8, 0.3],
        }
    )

    result = virtual_screen.add_consensus_columns(frame)

    assert result["consensus_all_antagonistic"].tolist() == [1, 0, 0]
    assert result["consensus_all_non_antagonistic"].tolist() == [0, 0, 1]
    assert result["consensus_vote_count"].tolist() == [3, 2, 0]


def test_lightgbm_feature_routing() -> None:
    combined = np.arange(2 * 2248).reshape(2, 2248)

    routed = virtual_screen.extract_lgbm_features(combined)

    assert routed.shape == (2, 1224)
    assert (routed[:, :1024] == combined[:, :1024]).all()
    assert (routed[:, 1024:] == combined[:, 2048:]).all()


def test_lightgbm_prediction_uses_fitted_feature_names() -> None:
    class NamedModel:
        feature_names_in_ = np.asarray([f"Column_{index}" for index in range(1224)])

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            assert isinstance(features, pd.DataFrame)
            assert features.columns.tolist() == self.feature_names_in_.tolist()
            return np.tile([0.25, 0.75], (len(features), 1))

    scores = virtual_screen.predict_lgbm_proba(
        NamedModel(), np.zeros((2, 2248), dtype=np.float32)
    )

    assert scores.tolist() == [0.75, 0.75]


def test_consensus_mean_score_ecdf_is_written(tmp_path: Path) -> None:
    sources = {
        name: pd.DataFrame({"consensus_mean_score": [0.1, 0.4, 0.9]})
        for name in ("gpcrdb", "zinc", "reinvent", "ouroboros")
    }

    output = virtual_screen._write_consensus_mean_score_ecdf(sources, tmp_path)

    assert output.name == "consensus_mean_score_ecdf.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_two_stage_selection_table_includes_all_families_and_top3(tmp_path: Path) -> None:
    families = [
        "LightGBM",
        "TabPFN",
        "RF",
        "XGB",
        "Chemprop",
        "KNN",
        "SVM",
        "LR",
    ]
    evidence = {
        "rows": [
            {
                "test_roc_rank": rank,
                "selected_for_screening": rank <= 3,
                "model_family": family,
                "candidate_id": family.casefold(),
                "model_type": f"{family} representation",
                "valid_roc_mean": 0.8 - rank / 100,
                "valid_roc_std": 0.05,
                "test_roc_mean": 0.75 - rank / 100,
                "test_roc_std": 0.04,
            }
            for rank, family in enumerate(families, start=1)
        ]
    }

    csv_path, html_path = virtual_screen._write_selection_table(evidence, tmp_path)
    table = pd.read_csv(csv_path)

    assert len(table) == 8
    assert "Chemprop" in table["Model family"].tolist()
    assert table["Screening ensemble"].fillna("—").tolist()[:3] == ["Top 3"] * 3
    normalized_html = " ".join(html_path.read_text(encoding="utf-8").split())
    assert "not independent validation" in normalized_html


def test_public_checksums_exclude_regenerable_model_artifacts(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("report\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    (models / "model_manifest.json").write_text("{}\n", encoding="utf-8")
    (models / "model.joblib").write_bytes(b"regenerable")

    virtual_screen._write_checksums(tmp_path)

    checksums = (tmp_path / "checksums.sha256").read_text(encoding="utf-8")
    assert "report.md" in checksums
    assert "model_manifest.json" in checksums
    assert ".joblib" not in checksums
