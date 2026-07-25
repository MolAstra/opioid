"""Leakage-safe Chemprop variants used beside the tabular benchmark matrix."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from opioid.benchmark.context import LABEL_COLUMN, SEED, SMILES_COLUMN


# Reviewer Chemprop runs are intentionally GPU-backed.  A single visible GPU
# keeps each fold/candidate trial deterministic with respect to the selected
# CUDA device while allowing the caller to choose it via CUDA_VISIBLE_DEVICES.
CHEMPROP_ACCELERATOR = "gpu"
CHEMPROP_DEVICES = "1"


@dataclass(frozen=True)
class ChempropSpec:
    """A fixed Chemprop feature variant, not part of the tabular matrix."""

    candidate_id: str
    display_name: str
    molecule_featurizers: tuple[str, ...]

    @property
    def family(self) -> str:
        return "chemprop"

    @property
    def algorithm(self) -> str:
        return "chemprop"

    @property
    def representation(self) -> str:
        return "+".join(self.molecule_featurizers) if self.molecule_featurizers else "graph"

    @property
    def preset(self) -> str:
        return self.candidate_id.removeprefix("chemprop_")


CHEMPROP_SPECS: tuple[ChempropSpec, ...] = (
    ChempropSpec("chemprop_default", "Chemprop", ()),
    ChempropSpec("chemprop_morgan", "Chemprop + Morgan", ("morgan_binary",)),
    ChempropSpec(
        "chemprop_morgan_rdkit2d",
        "Chemprop + Morgan + RDKit2D",
        ("morgan_binary", "v1_rdkit_2d_normalized"),
    ),
)


def chemprop_candidates() -> tuple[ChempropSpec, ...]:
    return CHEMPROP_SPECS


def chemprop_runtime_manifest() -> dict[str, str | None]:
    """Return the fixed hardware configuration used by reviewer Chemprop jobs."""

    return _runtime_manifest(CHEMPROP_ACCELERATOR, CHEMPROP_DEVICES)


class ChempropRunner:
    """CLI adapter that keeps each DPBench fixed test split out of training."""

    def __init__(
        self,
        *,
        seed: int = SEED,
        executable: str = "chemprop",
        accelerator: str = CHEMPROP_ACCELERATOR,
        devices: str = CHEMPROP_DEVICES,
    ) -> None:
        if not accelerator:
            raise ValueError("Chemprop accelerator must be specified")
        if not devices:
            raise ValueError("Chemprop devices must be specified")
        self.seed = seed
        self.executable = executable
        self.accelerator = accelerator
        self.devices = devices

    @property
    def runtime_manifest(self) -> dict[str, str | None]:
        """Describe the concrete CLI runtime settings for provenance."""

        return _runtime_manifest(self.accelerator, self.devices)

    @property
    def output_namespace(self) -> str:
        """Isolate checkpoints made with different hardware configurations."""

        accelerator = _path_token(self.accelerator)
        devices = _path_token(self.devices)
        visible_devices = _path_token(os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
        return f"{accelerator}_devices_{devices}_visible_{visible_devices}"

    def fit_fold(
        self,
        spec: ChempropSpec,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        output_dir: Path,
    ) -> list[Path]:
        return self._fit_with_explicit_validation(spec, train_df, valid_df, output_dir)

    def _fit_with_explicit_validation(
        self,
        spec: ChempropSpec,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        output_dir: Path,
    ) -> list[Path]:
        work_dir = output_dir / spec.candidate_id
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        input_path = data_dir / "train_valid.csv"
        combined = pd.concat(
            [
                _training_frame(train_df, split="train"),
                _training_frame(valid_df, split="val"),
            ],
            ignore_index=True,
        )
        combined.to_csv(input_path, index=False)

        command = [
            self.executable,
            "train",
            "-i",
            str(input_path),
            "-o",
            str(work_dir),
            "-t",
            "classification",
            "--splits-column",
            "__split",
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
            "--tracking-metric",
            "val_loss",
            "--target-columns",
            LABEL_COLUMN,
            "--smiles-columns",
            SMILES_COLUMN,
            "--accelerator",
            self.accelerator,
            "--devices",
            self.devices,
            "--data-seed",
            str(self.seed),
            "--pytorch-seed",
            str(self.seed),
        ]
        if spec.molecule_featurizers:
            command.extend(["--molecule-featurizers", *spec.molecule_featurizers])
        subprocess.run(command, check=True)

        checkpoints = sorted(work_dir.rglob("*.pt"))
        if not checkpoints:
            raise RuntimeError(f"No Chemprop checkpoints found under {work_dir}")
        return checkpoints

    def predict(
        self,
        spec: ChempropSpec,
        checkpoints: Sequence[Path],
        predict_df: pd.DataFrame,
        output_path: Path,
    ) -> np.ndarray:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_path = output_path.parent / f"{output_path.stem}_input.csv"
        predict_df[[SMILES_COLUMN, LABEL_COLUMN]].to_csv(input_path, index=False)
        command = [
            self.executable,
            "predict",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--model-paths",
            *(str(path) for path in checkpoints),
            "--smiles-columns",
            SMILES_COLUMN,
            "--accelerator",
            self.accelerator,
            "--devices",
            self.devices,
        ]
        if spec.molecule_featurizers:
            command.extend(["--molecule-featurizers", *spec.molecule_featurizers])
        subprocess.run(command, check=True)
        return _read_probabilities(output_path)


def _training_frame(df: pd.DataFrame, *, split: str) -> pd.DataFrame:
    work = df[[SMILES_COLUMN, LABEL_COLUMN]].copy()
    work["__split"] = split
    return work


def _read_probabilities(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return np.zeros((0,), dtype=np.float64)
    columns = [
        column for column in rows[0] if column != SMILES_COLUMN and "unc" not in column.lower()
    ]
    if not columns:
        raise RuntimeError(f"Could not identify Chemprop probability column in {path}")
    preferred = LABEL_COLUMN if LABEL_COLUMN in columns else columns[0]
    return np.asarray([float(row[preferred]) for row in rows], dtype=np.float64)


def _path_token(value: str) -> str:
    """Turn a CLI value into a stable, local-path-safe namespace component."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "default"


def _runtime_manifest(accelerator: str, devices: str) -> dict[str, str | None]:
    return {
        "accelerator": accelerator,
        "devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_selection": "CUDA_VISIBLE_DEVICES controls the visible GPU",
    }
