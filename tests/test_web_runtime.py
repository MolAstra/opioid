from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from opioid.screen.virtual_screen import LGBM_ID, RF_ID, TABPFN_ID
from opioid.web.consensus_runtime import ConsensusRuntime, WebArtifactPaths, _draw_fragment_svg


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_artifacts(tmp_path: Path) -> WebArtifactPaths:
    rf_bundle = tmp_path / "workflow" / "50_explain_rf" / "results"
    screening = tmp_path / "workflow" / "60_virtual_screen" / "results"
    (rf_bundle / "model").mkdir(parents=True)
    (screening / "models").mkdir(parents=True)

    rng = np.random.default_rng(42)
    combined = rng.integers(0, 2, size=(12, 2248)).astype(np.float32)
    labels = np.asarray([0, 1] * 6)
    rf = RandomForestClassifier(n_estimators=8, random_state=42).fit(combined[:, :2048], labels)
    lgbm_stand_in = SVC(probability=True, random_state=42).fit(
        np.concatenate((combined[:, :1024], combined[:, 2048:]), axis=1), labels
    )
    tabpfn_stand_in = SVC(probability=True, random_state=42).fit(
        combined[:, :2048], labels
    )

    rf_path = rf_bundle / "model" / "rf__ecfp_2048.joblib"
    lgbm_path = screening / "models" / f"{LGBM_ID}.joblib"
    tabpfn_path = screening / "models" / f"{TABPFN_ID}.joblib"
    joblib.dump(rf, rf_path)
    joblib.dump(lgbm_stand_in, lgbm_path)
    joblib.dump(tabpfn_stand_in, tabpfn_path)
    (rf_bundle / "model" / "model_manifest.json").write_text(
        json.dumps(
            {
                "artifact_path": "model/rf__ecfp_2048.joblib",
                "artifact_sha256": _sha256(rf_path),
                "candidate": {"candidate_id": RF_ID},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "bit": np.arange(2048),
            "mean_abs_shap": np.zeros(2048),
            "carrier_count": np.zeros(2048, dtype=int),
            "antagonistic_prevalence": np.zeros(2048),
            "non_antagonistic_prevalence": np.zeros(2048),
        }
    ).to_csv(rf_bundle / "bit_summary.csv", index=False)
    (screening / "run_manifest.json").write_text(
        json.dumps({"selection_scope": "test_informed_screening_ensemble"}),
        encoding="utf-8",
    )
    (screening / "models" / "model_manifest.json").write_text(
        json.dumps(
            {
                "models": {
                    RF_ID: {"artifact_path": str(rf_path), "artifact_sha256": _sha256(rf_path)},
                    LGBM_ID: {
                        "artifact_path": lgbm_path.name,
                        "artifact_sha256": _sha256(lgbm_path),
                    },
                    TABPFN_ID: {
                        "artifact_path": tabpfn_path.name,
                        "artifact_sha256": _sha256(tabpfn_path),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return WebArtifactPaths(tmp_path, rf_bundle, screening)


def test_consensus_runtime_predicts_and_explains(tmp_path: Path) -> None:
    runtime = ConsensusRuntime(_fake_artifacts(tmp_path))

    rows = runtime.predict_requests(
        [
            {"smiles": "CCO", "explain": True},
            {"smiles": "OCC", "explain": False},
            {"smiles": "not-smiles", "explain": True},
        ]
    )

    assert rows[0]["status"] == "ok"
    assert rows[0]["canonical_smiles"] == rows[1]["canonical_smiles"]
    assert rows[0]["consensus_vote_count"] in {0, 1, 2, 3}
    assert rows[0]["consensus_all_antagonistic"] == int(rows[0]["consensus_vote_count"] == 3)
    explanation = rows[0]["explanation"]
    assert explanation["scope"] == "rf_branch_only"
    assert explanation["additivity_residual"] < 1e-5
    assert explanation["structure_png_data_url"].startswith("data:image/png;base64,")
    assert explanation["structure_svg_data_url"].startswith("data:image/svg+xml;base64,")
    contributions = explanation["positive_contributions"] + explanation["negative_contributions"]
    for contribution in contributions:
        assert len(contribution["environment_svg_data_urls"]) == len(
            contribution["environment_smiles"]
        )
        assert all(
            item.startswith("data:image/svg+xml;base64,")
            for item in contribution["environment_svg_data_urls"]
        )
    assert rows[1]["explanation"] is None
    assert rows[2]["status"] == "invalid_smiles"
    assert rows[2]["rf_antagonistic_score"] is None
    assert len(runtime._prediction_cache) == 1


def test_consensus_runtime_rejects_changed_artifact(tmp_path: Path) -> None:
    paths = _fake_artifacts(tmp_path)
    artifact = paths.screening_bundle / "models" / f"{LGBM_ID}.joblib"
    artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        ConsensusRuntime(paths)


def test_consensus_runtime_requires_workflow_bundles(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workflow 50 and Workflow 60"):
        ConsensusRuntime(WebArtifactPaths.from_repo(tmp_path))


def test_fragment_renderer_supports_open_aromatic_environments() -> None:
    svg = _draw_fragment_svg("c(c)c")

    assert svg.startswith("<?xml")
    assert "<svg" in svg
