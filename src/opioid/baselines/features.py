from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from opioid.baselines import config

RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    rows: pd.DataFrame
    invalid_smiles: list[str]


def _morgan_array(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=config.MORGAN_RADIUS,
        nBits=config.MORGAN_BITS,
    )
    arr = np.zeros((config.MORGAN_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_dataset(df: pd.DataFrame) -> FeatureDataset:
    features: list[np.ndarray] = []
    keep_rows: list[int] = []
    invalid_smiles: list[str] = []

    smiles_values = df[config.SMILES_COLUMN].astype(str).tolist()
    for idx, smiles in enumerate(smiles_values):
        fp = _morgan_array(smiles)
        if fp is None:
            invalid_smiles.append(smiles)
            continue
        keep_rows.append(idx)
        features.append(fp)

    if features:
        x = np.stack(features, axis=0)
    else:
        x = np.zeros((0, config.MORGAN_BITS), dtype=np.float32)

    kept_df = df.iloc[keep_rows].reset_index(drop=True)
    y = kept_df[config.LABEL_COLUMN].to_numpy(dtype=np.int64)
    return FeatureDataset(features=x, labels=y, rows=kept_df, invalid_smiles=invalid_smiles)

