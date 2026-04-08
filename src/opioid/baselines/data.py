from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opioid.baselines import config


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_df: pd.DataFrame
    valid_df: pd.DataFrame


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {config.SMILES_COLUMN, config.LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df[[config.SMILES_COLUMN, config.LABEL_COLUMN]].copy()


def load_test_split() -> pd.DataFrame:
    return _load_csv(config.INPUT_TEST_CSV)


def load_fold_splits() -> list[FoldSplit]:
    splits: list[FoldSplit] = []
    for fold_index in range(config.N_FOLDS):
        fold_dir = config.FOLD_DIR / f"fold_{fold_index}"
        splits.append(
            FoldSplit(
                fold_index=fold_index,
                train_df=_load_csv(fold_dir / "train.csv"),
                valid_df=_load_csv(fold_dir / "valid.csv"),
            )
        )
    return splits


def load_development_set() -> pd.DataFrame:
    frames = []
    for split in load_fold_splits():
        frames.append(split.train_df)
        frames.append(split.valid_df)

    dev_df = pd.concat(frames, ignore_index=True)
    return dev_df.drop_duplicates().reset_index(drop=True)

