"""Validated access to persisted protocol split tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opioid.benchmark.context import LABEL_COLUMN, N_FOLDS, SMILES_COLUMN, RunContext


@dataclass(frozen=True)
class FoldSplit:
    """One persisted train/validation fold."""

    fold_index: int
    train_df: pd.DataFrame
    valid_df: pd.DataFrame


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing persisted split: {path}")
    df = pd.read_csv(path)
    required = {SMILES_COLUMN, LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path} is empty")
    work = df.copy()
    work[SMILES_COLUMN] = work[SMILES_COLUMN].astype(str)
    work[LABEL_COLUMN] = pd.to_numeric(work[LABEL_COLUMN], errors="raise").astype(int)
    if not work[LABEL_COLUMN].isin((0, 1)).all():
        raise ValueError(f"{path} must contain binary labels 0 and 1")
    return work


def load_test_split(context: RunContext) -> pd.DataFrame:
    return _load_csv(context.split_dir / "test.csv")


def load_fold_splits(context: RunContext) -> list[FoldSplit]:
    folds: list[FoldSplit] = []
    for fold_index in range(N_FOLDS):
        train_path, valid_path = _fold_paths(context.split_dir, fold_index)
        folds.append(
            FoldSplit(
                fold_index=fold_index,
                train_df=_load_csv(train_path),
                valid_df=_load_csv(valid_path),
            )
        )
    return folds


def load_development_set(context: RunContext) -> pd.DataFrame:
    """Return the unique development set without ever reading the test table."""

    folds = load_fold_splits(context)
    frames = [split.train_df for split in folds]
    frames.extend(split.valid_df for split in folds)
    development = pd.concat(frames, ignore_index=True)
    subset = ["sample_id"] if "sample_id" in development.columns else None
    return development.drop_duplicates(subset=subset).reset_index(drop=True)


def _fold_paths(split_dir: Path, fold_index: int) -> tuple[Path, Path]:
    """Read DPBench local-partition files while retaining legacy fixture support."""

    dpbench = (
        split_dir / f"train_fold-{fold_index}.csv",
        split_dir / f"valid_fold-{fold_index}.csv",
    )
    if dpbench[0].exists() or dpbench[1].exists():
        return dpbench
    legacy_dir = split_dir / f"fold_{fold_index}"
    return legacy_dir / "train.csv", legacy_dir / "valid.csv"
