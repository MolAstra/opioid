from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from opioid.baselines import config


def save_classic_model(model, model_name: str, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    return path


def write_runtime_config(
    artifacts_dir: Path,
    model_paths: dict[str, Path],
    top_models: list[str],
    best_model: str,
) -> Path:
    payload = {
        "smiles_col": config.SMILES_COLUMN,
        "label_col": config.LABEL_COLUMN,
        "threshold": config.THRESHOLD,
        "feature": {"type": "morgan", "radius": config.MORGAN_RADIUS, "n_bits": config.MORGAN_BITS},
        "all_models": sorted(model_paths),
        "top_models": top_models,
        "best_model": best_model,
        "model_files": {name: str(path) for name, path in sorted(model_paths.items())},
    }
    path = artifacts_dir / "runtime_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_rank_outputs(cv_summary: pd.DataFrame, output_dir: Path) -> tuple[list[str], str]:
    leaderboard = cv_summary.sort_values(
        [
            "cv_test_roc_mean",
            "test_roc",
            "valid_roc_mean",
            "cv_test_prc_mean",
            "test_prc",
            "valid_prc_mean",
            "cv_test_f1_mean",
            "test_f1",
            "valid_f1_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)

    top3 = leaderboard.head(3).copy()
    top3_models = top3["model_label"].tolist()
    top1_model = top3_models[0]

    (output_dir / "top3_models.json").write_text(
        json.dumps(
            {
                "selection_rule": "top-k by cv_test > final_test > cv_valid, using roc then prc then f1",
                "top_models": top3[
                    [
                        "model_label",
                        "cv_test_roc_mean",
                        "test_roc",
                        "valid_roc_mean",
                        "cv_test_prc_mean",
                        "test_prc",
                        "valid_prc_mean",
                        "cv_test_f1_mean",
                        "test_f1",
                        "valid_f1_mean",
                    ]
                ].to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "top1_model.json").write_text(
        json.dumps({"best_model": top1_model, "selection_rule": "cv_test > final_test > cv_valid"}, indent=2),
        encoding="utf-8",
    )

    return top3_models, top1_model
