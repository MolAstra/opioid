from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from opioid.baselines import config
from opioid.baselines.features import build_feature_dataset
from opioid.baselines.models import predict_probabilities


class DeployableFamilyPredictor:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.runtime = json.loads((self.artifacts_dir / "runtime_config.json").read_text(encoding="utf-8"))
        self.all_models = list(self.runtime["all_models"])
        self.top_models = list(self.runtime["top_models"])
        self.best_model = str(self.runtime["best_model"])
        self.threshold = float(self.runtime["threshold"])
        model_files = dict(self.runtime["model_files"])
        self.models = {
            name: joblib.load(Path(model_files[name]))
            for name in self.all_models
            if str(model_files[name]).endswith(".joblib")
        }

    def predict_many(self, smiles_list: list[str]) -> dict:
        df = pd.DataFrame({config.SMILES_COLUMN: smiles_list, config.LABEL_COLUMN: 0})
        dataset = build_feature_dataset(df)
        all_probabilities = {
            name: predict_probabilities(model, dataset.features)
            for name, model in self.models.items()
        }

        rows = []
        for row_index, (_, row) in enumerate(dataset.rows.iterrows()):
            result = {config.SMILES_COLUMN: row[config.SMILES_COLUMN]}
            all_labels = []
            top_labels = []
            for name in self.all_models:
                prob = float(all_probabilities[name][row_index])
                label = int(prob >= self.threshold)
                result[f"{name}_prob"] = prob
                result[f"{name}_label"] = label
                all_labels.append(label)
                if name in self.top_models:
                    top_labels.append(label)

            result["all_models_all1"] = int(all(value == 1 for value in all_labels))
            result["all_models_all0"] = int(all(value == 0 for value in all_labels))
            result["top3_all1"] = int(len(top_labels) > 0 and all(value == 1 for value in top_labels))
            result["top3_all0"] = int(len(top_labels) > 0 and all(value == 0 for value in top_labels))
            rows.append(result)

        return {
            "rows": rows,
            "best_model": self.best_model,
            "top_models": self.top_models,
            "all_models": self.all_models,
            "invalid_smiles": dataset.invalid_smiles,
        }
