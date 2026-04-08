from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from opioid.baselines import config


def ensure_output_dirs() -> None:
    for path in [
        config.OUTPUT_DIR,
        config.ARTIFACTS_DIR,
        config.TEST_PREDICTIONS_DIR,
        config.CHEMPROP_WORK_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_test_predictions(model_label: str, rows: pd.DataFrame, y_prob: np.ndarray) -> Path:
    out = rows[[config.SMILES_COLUMN, config.LABEL_COLUMN]].copy()
    out["pred_prob"] = y_prob
    out["pred_label"] = (y_prob >= config.THRESHOLD).astype(int)
    path = config.TEST_PREDICTIONS_DIR / f"{model_label}.csv"
    out.to_csv(path, index=False)
    return path


def write_run_config(payload: dict) -> Path:
    path = config.OUTPUT_DIR / "run_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

