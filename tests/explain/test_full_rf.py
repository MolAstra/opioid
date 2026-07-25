from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier

from opioid.chemistry.representations import FeatureDataset, get_feature_spec
from opioid.explain import full_rf


def _write_protocol(task_root: Path, protocol: str, *, mutate_last_label: bool = False) -> None:
    root = task_root / f"processed_{protocol}"
    root.mkdir(parents=True)
    labels = np.asarray([0] * 754 + [1] * 228, dtype=int)
    if mutate_last_label:
        labels[-1] = 0
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(982),
            "smiles": ["C"] * 982,
            "label": labels,
        }
    )
    frame.iloc[:700].to_csv(root / "train_fold-0.csv", index=False)
    frame.iloc[700:900].to_csv(root / "valid_fold-0.csv", index=False)
    frame.iloc[900:].to_csv(root / "test.csv", index=False)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")


def _write_task(task_root: Path) -> None:
    for protocol in full_rf.PROTOCOLS:
        _write_protocol(task_root, protocol)


def test_load_full_corpus_requires_identical_protocol_unions(tmp_path: Path) -> None:
    _write_task(tmp_path)

    corpus = full_rf.load_full_corpus(tmp_path)

    assert len(corpus.frame) == 982
    assert corpus.frame["label"].value_counts().sort_index().to_dict() == {0: 754, 1: 228}
    assert len(corpus.input_files) == 12

    changed_root = tmp_path / "changed"
    for protocol in ("scaffold", "random"):
        _write_protocol(changed_root, protocol)
    _write_protocol(changed_root, "datasail", mutate_last_label=True)
    with pytest.raises(ValueError, match="same canonical labeled corpus"):
        full_rf.load_full_corpus(changed_root)


def test_compute_class1_shap_is_additive() -> None:
    rng = np.random.default_rng(42)
    features = rng.integers(0, 2, size=(24, 2048), dtype=np.int8)
    labels = np.asarray([0, 1] * 12)
    model = RandomForestClassifier(n_estimators=8, random_state=42).fit(features, labels)

    values, expected = full_rf.compute_class1_shap(model, features)

    assert values.shape == features.shape
    np.testing.assert_allclose(
        expected + values.sum(axis=1),
        model.predict_proba(features)[:, 1],
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_cases_are_deterministic_and_disjoint() -> None:
    rng = np.random.default_rng(7)
    rows = pd.DataFrame(
        {
            "sample_id": [str(index) for index in range(30)],
            "smiles": ["C"] * 30,
            "label": [0] * 15 + [1] * 15,
        }
    )
    dataset = FeatureDataset(
        features=rng.integers(0, 2, size=(30, 32), dtype=np.int8).astype(np.float32),
        labels=rows["label"].to_numpy(),
        rows=rows,
        invalid_smiles=[],
        feature_spec=get_feature_spec("ecfp_2048"),
        source_row_indices=list(range(30)),
        invalid_row_indices=[],
        canonical_smiles=["C"] * 30,
    )
    probabilities = np.linspace(0.02, 0.98, 30)

    first = full_rf._select_local_cases(dataset, probabilities)
    second = full_rf._select_local_cases(dataset, probabilities)

    pd.testing.assert_frame_equal(first, second)
    assert first["sample_id"].is_unique
    assert first.groupby("category").size().to_dict() == {
        "boundary": 4,
        "high_antagonistic": 4,
        "high_non_antagonistic": 4,
    }


def test_load_full_rf_bundle_validates_artifact_hash(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "rf__ecfp_2048.joblib"
    model = RandomForestClassifier(n_estimators=1, random_state=42).fit([[0], [1]], [0, 1])
    joblib.dump(model, model_path)
    manifest = {
        "schema_version": full_rf.FULL_RF_SCHEMA_VERSION,
        "candidate": {"candidate_id": full_rf.MODEL_CANDIDATE_ID},
        "training_scope": "full_deduplicated_labeled_corpus",
        "independent_evaluation": False,
        "artifact_path": "model/rf__ecfp_2048.joblib",
        "artifact_sha256": full_rf._sha256(model_path),
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded, payload = full_rf.load_full_rf_bundle(tmp_path)

    assert loaded.predict([[1]]).tolist() == [1]
    assert payload["candidate"]["candidate_id"] == full_rf.MODEL_CANDIDATE_ID
    model_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        full_rf.load_full_rf_bundle(tmp_path)


def test_package_manifest_separates_runtime_model_from_published_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "report.md").write_text("report\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_manifest.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "rf__ecfp_2048.joblib").write_bytes(b"regenerable")

    manifest = full_rf._package_manifest(tmp_path)

    assert "report.md" in manifest["files"]
    assert "model/model_manifest.json" in manifest["files"]
    assert "model/rf__ecfp_2048.joblib" not in manifest["files"]
    assert (
        manifest["runtime_artifacts"]["model/rf__ecfp_2048.joblib"]["published"]
        is False
    )


def test_highlight_renderer_preserves_positive_and_negative_colors() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None

    image = full_rf._draw_highlighted_molecule(
        molecule,
        atom_colors={0: (0.20, 0.72, 0.33), 2: (0.84, 0.22, 0.18)},
        bond_colors={},
    )
    pixels = np.asarray(image.convert("RGB"), dtype=int)
    green = (pixels[:, :, 1] > pixels[:, :, 0] + 15) & (pixels[:, :, 1] > pixels[:, :, 2] + 15)
    red = (pixels[:, :, 0] > pixels[:, :, 1] + 15) & (pixels[:, :, 0] > pixels[:, :, 2] + 15)

    assert green.any()
    assert red.any()
