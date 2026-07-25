from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.pipeline import Pipeline

from opioid.benchmark import build_estimator, traditional_candidates
from opioid.chemistry import FEATURE_SPECS, build_feature_dataset, featurize_smiles
from opioid.chemistry import representations


class _DescriptorGenerator:
    def process(self, smiles: str) -> list[float | bool]:
        return [True, np.nan, np.inf, -np.inf, *([0.25] * 197)]


@pytest.fixture(autouse=True)
def _clear_feature_caches() -> None:
    representations._cached_feature_vector.cache_clear()
    yield
    representations._cached_feature_vector.cache_clear()


def test_registered_representations_have_expected_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(representations, "_descriptor_generator", lambda: _DescriptorGenerator())

    expected_widths = {
        "ecfp_1024": 1024,
        "ecfp_2048": 2048,
        "rdkit2d_normalized_200": 200,
        "ecfp_1024_rdkit2d_normalized_200": 1224,
        "ecfp_2048_rdkit2d_normalized_200": 2248,
    }
    for spec_id, expected_width in expected_widths.items():
        vector = featurize_smiles("OCC", FEATURE_SPECS[spec_id])
        assert vector.shape == (expected_width,)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()


def test_feature_dataset_retains_rows_labels_and_invalid_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(representations, "_descriptor_generator", lambda: _DescriptorGenerator())
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "not-a-smiles", "C[C@H](O)Cl"],
            "label": [1, 0, 0],
        },
        index=[10, 11, 12],
    )

    batch = build_feature_dataset(frame, "ecfp_2048_rdkit2d_normalized_200")

    assert batch.features.shape == (2, 2248)
    assert batch.labels is not None and batch.labels.tolist() == [1, 0]
    assert batch.rows["smiles"].tolist() == ["CCO", "C[C@H](O)Cl"]
    assert batch.invalid_smiles == ["not-a-smiles"]
    assert batch.source_row_indices == [10, 12]
    assert batch.invalid_row_indices == [11]


def test_descriptor_contract_rejects_bad_status_or_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedStatus:
        def process(self, smiles: str) -> list[float | bool]:
            return [False, *([0.0] * 200)]

    monkeypatch.setattr(representations, "_descriptor_generator", lambda: FailedStatus())
    with pytest.raises(representations.DescriptorGenerationError, match="unsuccessful status"):
        representations._normalized_descriptors("CCO")

    class WrongWidth:
        def process(self, smiles: str) -> list[float | bool]:
            return [True, *([0.0] * 199)]

    monkeypatch.setattr(representations, "_descriptor_generator", lambda: WrongWidth())
    with pytest.raises(representations.DescriptorGenerationError, match="exactly 200"):
        representations._normalized_descriptors("CCO")


def test_traditional_catalog_is_a_55_candidate_matrix() -> None:
    candidates = traditional_candidates()

    assert len(candidates) == 55
    assert len({candidate.candidate_id for candidate in candidates}) == 55
    assert {candidate.algorithm_id for candidate in candidates} == {
        "svm",
        "rf",
        "xgb",
        "lgbm",
        "tabpfn",
        "knn_k3",
        "knn_k5",
        "logreg_l1_c0_1",
        "logreg_l1_c1",
        "logreg_l2_c0_1",
        "logreg_l2_c1",
    }

    scaled_ids = {"svm", "knn_k3", "logreg_l1_c0_1"}
    for algorithm_id in scaled_ids:
        candidate = next(item for item in candidates if item.algorithm_id == algorithm_id)
        estimator = build_estimator(candidate)
        assert isinstance(estimator, Pipeline)
        assert list(estimator.named_steps) == ["scaler", "model"]

    rf_candidate = next(item for item in candidates if item.algorithm_id == "rf")
    assert not isinstance(build_estimator(rf_candidate), Pipeline)


def test_ecfp_2048_matches_the_morgan_bit_mapping_used_for_rf_shap() -> None:
    smiles = "CCOC(=O)c1ccccc1"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    shap_fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    expected = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(shap_fingerprint, expected)

    np.testing.assert_array_equal(featurize_smiles(smiles, "ecfp_2048"), expected)
