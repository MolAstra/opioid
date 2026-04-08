from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from rdkit import RDLogger

from opioid.baselines import config
from opioid.baselines.features import build_feature_dataset
from opioid.baselines.models import predict_probabilities

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

MAX_CSV_ROWS = 1000
ALLOWED_SMILES_COLUMNS = ("smiles", "SMILES")
PREDICTION_PROB_COL = "antagonist_probability"
PREDICTION_LABEL_COL = "pred_label"
STATUS_COL = "status"


class RFWebPredictor:
    def __init__(self, artifacts_dir: Path | str = config.ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        runtime_path = self.artifacts_dir / "runtime_config.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.threshold = float(runtime["threshold"])
        self.smiles_col = str(runtime["smiles_col"])
        best_model = str(runtime["best_model"])
        model_path = Path(runtime["model_files"][best_model])
        self.model_name = best_model
        self.model = joblib.load(model_path)

    def predict_single(self, smiles: str) -> pd.DataFrame:
        text = str(smiles).strip()
        if not text:
            raise ValueError("SMILES is required.")

        input_df = pd.DataFrame(
            {
                self.smiles_col: [text],
                config.LABEL_COLUMN: [0],
            }
        )
        dataset = build_feature_dataset(input_df)
        if dataset.invalid_smiles:
            raise ValueError(f"Invalid SMILES: {text}")

        prob = float(predict_probabilities(self.model, dataset.features)[0])
        return pd.DataFrame(
            [
                {
                    self.smiles_col: text,
                    PREDICTION_PROB_COL: prob,
                    PREDICTION_LABEL_COL: int(prob >= self.threshold),
                }
            ]
        )

    def predict_csv(self, csv_path: str | Path) -> tuple[pd.DataFrame, str]:
        path = Path(csv_path)
        input_df = pd.read_csv(path)
        if len(input_df) > MAX_CSV_ROWS:
            raise ValueError(f"CSV must contain at most {MAX_CSV_ROWS} rows.")

        smiles_col = self._find_smiles_column(input_df)
        result_df = input_df.copy()
        result_df[smiles_col] = result_df[smiles_col].astype(str)
        result_df[PREDICTION_PROB_COL] = pd.NA
        result_df[PREDICTION_LABEL_COL] = pd.NA
        result_df[STATUS_COL] = "ok"

        working_df = pd.DataFrame(
            {
                "__row_id": range(len(result_df)),
                self.smiles_col: result_df[smiles_col],
                config.LABEL_COLUMN: 0,
            }
        )
        dataset = build_feature_dataset(working_df)
        valid_indices = set(dataset.rows["__row_id"].tolist())
        invalid_indices = [idx for idx in range(len(result_df)) if idx not in valid_indices]
        if invalid_indices:
            result_df.loc[invalid_indices, STATUS_COL] = "invalid_smiles"

        if len(dataset.rows) > 0:
            probs = predict_probabilities(self.model, dataset.features)
            labels = (probs >= self.threshold).astype(int)
            row_ids = dataset.rows["__row_id"].to_numpy(dtype=int)
            result_df.loc[row_ids, PREDICTION_PROB_COL] = probs
            result_df.loc[row_ids, PREDICTION_LABEL_COL] = labels

        output_path = self._write_temp_csv(result_df)
        return result_df, output_path

    def _find_smiles_column(self, df: pd.DataFrame) -> str:
        for column in ALLOWED_SMILES_COLUMNS:
            if column in df.columns:
                return column
        raise ValueError("CSV must include a 'smiles' or 'SMILES' column.")

    def _write_temp_csv(self, df: pd.DataFrame) -> str:
        handle = tempfile.NamedTemporaryFile(prefix="opioid_rf_predictions_", suffix=".csv", delete=False)
        path = Path(handle.name)
        handle.close()
        df.to_csv(path, index=False)
        return str(path)
