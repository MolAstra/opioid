from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from opioid.baselines import config
from opioid.baselines.chemprop_utils import ChempropModel
from opioid.baselines.features import FeatureDataset


class ProbabilisticModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "ProbabilisticModel": ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


def classic_model_names() -> list[str]:
    return ["svm", "rf", "xgb", "lgbm", "tabpfn"]


def build_classic_model(name: str):
    if name == "svm":
        return SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=config.SEED)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced_subsample",
            random_state=config.SEED,
            n_jobs=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.SEED,
            n_jobs=4,
        )
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.SEED,
            n_jobs=4,
        )
    if name == "tabpfn":
        return TabPFNClassifier(
            device="auto",
            random_state=config.SEED,
            n_estimators=8,
            ignore_pretraining_limits=True,
        )
    raise ValueError(f"Unsupported classic model: {name}")


def predict_probabilities(model: ProbabilisticModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def train_chemprop_variant(
    model_label: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, Path]:
    variant_to_featurizers = {
        "chemprop_default": [],
        "chemprop_morgan": ["morgan_binary"],
        "chemprop_morgan_rdkit2d": ["morgan_binary", "v1_rdkit_2d_normalized"],
    }
    if model_label not in variant_to_featurizers:
        raise ValueError(f"Unsupported Chemprop model label: {model_label}")

    runner = ChempropModel(
        model_label=model_label,
        output_dir=output_dir,
        molecule_featurizers=variant_to_featurizers[model_label],
    )
    checkpoints = runner.fit(train_df=train_df, valid_df=valid_df, test_df=test_df)
    return runner.predict(
        checkpoint_paths=checkpoints,
        predict_df=test_df,
        preds_path=output_dir / model_label / "predictions.csv",
    )


def fit_chemprop_variant(
    model_label: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[ChempropModel, list[Path]]:
    variant_to_featurizers = {
        "chemprop_default": [],
        "chemprop_morgan": ["morgan_binary"],
        "chemprop_morgan_rdkit2d": ["morgan_binary", "v1_rdkit_2d_normalized"],
    }
    if model_label not in variant_to_featurizers:
        raise ValueError(f"Unsupported Chemprop model label: {model_label}")

    runner = ChempropModel(
        model_label=model_label,
        output_dir=output_dir,
        molecule_featurizers=variant_to_featurizers[model_label],
    )
    checkpoints = runner.fit(train_df=train_df, valid_df=valid_df, test_df=test_df)
    return runner, checkpoints


def train_classic_and_predict(
    model_name: str,
    train_data: FeatureDataset,
    test_data: FeatureDataset,
):
    model = build_classic_model(model_name)
    model.fit(train_data.features, train_data.labels)
    return model, predict_probabilities(model, test_data.features)
