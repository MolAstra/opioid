from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from opioid.baselines import config


class ChempropModel:
    def __init__(self, model_label: str, output_dir: Path, molecule_featurizers: list[str]) -> None:
        self.model_label = model_label
        self.output_dir = output_dir
        self.molecule_featurizers = molecule_featurizers

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ) -> list[Path]:
        work_dir = self.output_dir / self.model_label
        data_dir = work_dir / "data"
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"

        data_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        data_paths = [str(train_path)]
        split_sizes = ["0.9", "0.1", "0.0"]
        if valid_df is not None:
            valid_path = data_dir / "valid.csv"
            valid_df.to_csv(valid_path, index=False)
            data_paths.append(str(valid_path))
            split_sizes = None
        data_paths.append(str(test_path))

        train_cmd = [
            "chemprop",
            "train",
            "-i",
            *data_paths,
            "-o",
            str(work_dir),
            "-t",
            "classification",
            "--epochs",
            "30",
            "--warmup-epochs",
            "2",
            "--patience",
            "5",
            "--metrics",
            "roc",
            "prc",
            "accuracy",
            "f1",
            "--target-columns",
            config.LABEL_COLUMN,
            "--smiles-columns",
            config.SMILES_COLUMN,
            "--accelerator",
            "cpu",
            "--devices",
            "auto",
            "--data-seed",
            str(config.SEED),
            "--pytorch-seed",
            str(config.SEED),
        ]
        if split_sizes is not None:
            train_cmd.extend(["--split-sizes", *split_sizes])
        if self.molecule_featurizers:
            train_cmd.extend(["--molecule-featurizers", *self.molecule_featurizers])

        subprocess.run(train_cmd, check=True)

        checkpoint_paths = sorted(work_dir.rglob("*.pt"))
        if not checkpoint_paths:
            raise RuntimeError(f"No Chemprop checkpoints found under {work_dir}")
        return checkpoint_paths

    def predict(self, checkpoint_paths: list[Path], predict_df: pd.DataFrame, preds_path: Path) -> tuple[np.ndarray, Path]:
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        predict_input_path = preds_path.parent / f"{preds_path.stem}_input.csv"
        predict_df.to_csv(predict_input_path, index=False)
        predict_cmd = [
            "chemprop",
            "predict",
            "-i",
            str(predict_input_path),
            "-o",
            str(preds_path),
            "--model-paths",
            *(str(path) for path in checkpoint_paths),
            "--smiles-columns",
            config.SMILES_COLUMN,
            "--accelerator",
            "cpu",
            "--devices",
            "auto",
        ]
        if self.molecule_featurizers:
            predict_cmd.extend(["--molecule-featurizers", *self.molecule_featurizers])

        subprocess.run(predict_cmd, check=True)
        return _read_prediction_probabilities(preds_path), preds_path


def _read_prediction_probabilities(preds_path: Path) -> np.ndarray:
    with preds_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return np.zeros((0,), dtype=np.float64)

    if config.LABEL_COLUMN in rows[0]:
        return np.asarray([float(row[config.LABEL_COLUMN]) for row in rows], dtype=np.float64)

    candidate_columns = [name for name in rows[0] if name != config.SMILES_COLUMN]
    if not candidate_columns:
        raise RuntimeError(f"Could not find prediction columns in {preds_path}")
    preferred = [name for name in candidate_columns if "unc" not in name.lower()]
    score_column = preferred[0] if preferred else candidate_columns[0]
    return np.asarray([float(row[score_column]) for row in rows], dtype=np.float64)
