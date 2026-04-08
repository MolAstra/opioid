from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from opioid.baselines import config
from opioid.screen.candidates import TopModelPredictor, _load_training_smiles

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

OUTPUT_DIR = Path("outputs/novel_mols")
BACKUP_DIR = Path("outputs/tmp_novel_mols")
INPUT_RAW_CSV = OUTPUT_DIR / "external_mor_20_raw.csv"
INPUT_CLEAN_CSV = OUTPUT_DIR / "external_mor_20.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "external_top3_predictions.csv"
SUMMARY_CSV = OUTPUT_DIR / "external_top3_summary.csv"
ANALYSIS_MD = OUTPUT_DIR / "external_top3_analysis.md"
INVALID_SMILES_JSON = OUTPUT_DIR / "external_invalid_smiles.json"
OVERLAP_REPORT_CSV = OUTPUT_DIR / "external_overlap_report.csv"

REQUIRED_COLUMNS = ["smiles", "label", "name", "id", "doi"]


def _canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_and_clean_raw() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    raw_df = pd.read_csv(INPUT_RAW_CSV)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {INPUT_RAW_CSV}: {missing_columns}")

    work_df = raw_df.copy()
    work_df["smiles"] = work_df["smiles"].astype(str)
    work_df["canonical_smiles"] = work_df["smiles"].map(_canonicalize_smiles)

    invalid_mask = work_df["canonical_smiles"].isna()
    invalid_smiles = work_df.loc[invalid_mask, "smiles"].tolist()
    valid_df = work_df.loc[~invalid_mask].copy()

    overlap_set = _load_training_smiles()
    overlap_mask = valid_df["smiles"].isin(overlap_set) | valid_df["canonical_smiles"].isin(overlap_set)
    overlap_df = valid_df.loc[overlap_mask, ["id", "name", "smiles", "doi"]].copy()
    filtered_df = valid_df.loc[~overlap_mask].copy()

    filtered_df = (
        filtered_df.sort_values(by=["id", "name"])
        .drop_duplicates(subset=["canonical_smiles"], keep="first")
        .reset_index(drop=True)
    )
    filtered_df["smiles"] = filtered_df["canonical_smiles"]
    cleaned = filtered_df[REQUIRED_COLUMNS].copy()
    return cleaned, invalid_smiles, overlap_df


def _metric_row(name: str, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, float | int | str]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {
        "predictor": name,
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _build_summary_df(pred_df: pd.DataFrame, top_models: list[str]) -> pd.DataFrame:
    y_true = pred_df["label"].astype(int)
    rows = []
    for model_name in top_models:
        rows.append(
            _metric_row(
                name=model_name,
                y_true=y_true,
                y_pred=pred_df[f"{model_name}_label"].astype(int),
                y_prob=pred_df[f"{model_name}_prob"].astype(float),
            )
        )
    rows.append(
        _metric_row(
            name="top3_consensus",
            y_true=y_true,
            y_pred=pred_df["top3_all1"].astype(int),
            y_prob=pred_df["top3_mean_prob"].astype(float),
        )
    )
    return pd.DataFrame(rows)


def _build_analysis_text(
    pred_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_models: list[str],
    n_raw: int,
    n_invalid_smiles: int,
    n_overlap_removed: int,
) -> str:
    top_hits = pred_df.sort_values("top3_mean_prob", ascending=False).head(10)[
        ["id", "name", "label", "doi", "top3_mean_prob", "top3_min_prob", "top3_all1"]
    ]
    false_positives = pred_df[
        (pred_df["label"] == 0) & (pred_df["top3_all1"] == 1)
    ].sort_values("top3_mean_prob", ascending=False).head(5)[
        ["id", "name", "doi", "top3_mean_prob", "top3_all1"]
    ]
    false_negatives = pred_df[
        (pred_df["label"] == 1) & (pred_df["top3_all1"] == 0)
    ].sort_values("top3_mean_prob", ascending=True).head(5)[
        ["id", "name", "doi", "top3_mean_prob", "top3_all1"]
    ]

    lines = [
        "# External MOR 20 Top3 Prediction Analysis",
        "",
        f"- Top3 models: {', '.join(top_models)}",
        f"- Raw molecules: {n_raw}",
        f"- Invalid SMILES removed: {n_invalid_smiles}",
        f"- Overlap with train/valid/test removed: {n_overlap_removed}",
        f"- Final molecules predicted: {len(pred_df)}",
        f"- Positive labels: {int((pred_df['label'] == 1).sum())}",
        f"- Negative labels: {int((pred_df['label'] == 0).sum())}",
        "",
        "## Model Metrics",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Top Ranked Molecules",
        "",
        top_hits.to_markdown(index=False),
        "",
        "## Top False Positives by top3_mean_prob",
        "",
        false_positives.to_markdown(index=False),
        "",
        "## Top False Negatives by top3_mean_prob",
        "",
        false_negatives.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def run_novel_molecule_prediction() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = TopModelPredictor(config.ARTIFACTS_DIR / "runtime_config.json")

    cleaned_df, invalid_smiles, overlap_df = _load_and_clean_raw()
    cleaned_df.to_csv(INPUT_CLEAN_CSV, index=False)
    overlap_df.to_csv(OVERLAP_REPORT_CSV, index=False)
    INVALID_SMILES_JSON.write_text(json.dumps({"invalid_smiles": invalid_smiles}, indent=2), encoding="utf-8")

    pred_df, _ = predictor.predict_dataframe(cleaned_df)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    summary_df = _build_summary_df(pred_df=pred_df, top_models=predictor.top_models)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    n_raw = int(pd.read_csv(INPUT_RAW_CSV).shape[0])
    ANALYSIS_MD.write_text(
        _build_analysis_text(
            pred_df=pred_df,
            summary_df=summary_df,
            top_models=predictor.top_models,
            n_raw=n_raw,
            n_invalid_smiles=len(invalid_smiles),
            n_overlap_removed=len(overlap_df),
        ),
        encoding="utf-8",
    )

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        INPUT_RAW_CSV,
        INPUT_CLEAN_CSV,
        PREDICTIONS_CSV,
        SUMMARY_CSV,
        ANALYSIS_MD,
        OVERLAP_REPORT_CSV,
        INVALID_SMILES_JSON,
    ]:
        shutil.copy2(path, BACKUP_DIR / path.name)

    return {
        "raw_csv": INPUT_RAW_CSV,
        "clean_csv": INPUT_CLEAN_CSV,
        "predictions_csv": PREDICTIONS_CSV,
        "summary_csv": SUMMARY_CSV,
        "analysis_md": ANALYSIS_MD,
        "overlap_report_csv": OVERLAP_REPORT_CSV,
        "invalid_smiles_json": INVALID_SMILES_JSON,
    }
