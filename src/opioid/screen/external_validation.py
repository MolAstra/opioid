"""Exploratory literature-case evaluation for the fixed screening ensemble."""

from __future__ import annotations

import hashlib
import html
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from opioid.chemistry.representations import (
    FeatureRepresentationError,
    canonicalize_smiles,
)
from opioid.explain.full_rf import load_full_corpus
from opioid.web.consensus_runtime import ConsensusRuntime, WebArtifactPaths


SCHEMA_VERSION = 1
EXPECTED_ROWS = 20
EXPECTED_CLASS_COUNTS = {0: 10, 1: 10}
EXPECTED_OVERLAP_IDS = {"EXT-N05", "EXT-N08", "EXT-N10"}
PREDICTORS = (
    ("Random Forest", "rf_pred_label"),
    ("TabPFN", "tabpfn_pred_label"),
    ("LightGBM", "lightgbm_pred_label"),
    ("Strict three-model consensus", "strict_consensus_pred_label"),
)


@dataclass(frozen=True)
class ExternalValidationBundle:
    """Paths produced by one complete external-case run."""

    root: Path
    predictions_path: Path
    metrics_path: Path
    report_path: Path
    manifest_path: Path


def load_external_source(path: Path | str) -> pd.DataFrame:
    """Load and validate the fixed 20-compound literature source table."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing external-case source: {source_path}")
    frame = pd.read_csv(source_path)
    required = {"id", "name", "smiles", "label", "doi"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"External-case source is missing columns: {missing}")
    frame = frame[["id", "name", "smiles", "label", "doi"]].copy()
    for column in ("id", "name", "smiles", "doi"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"External-case source contains an empty {column}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("External-case labels must be binary 0/1")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} external cases, found {len(frame)}")
    class_counts = frame["label"].value_counts().sort_index().to_dict()
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"External-case class contract changed: {class_counts}")
    if frame["id"].duplicated().any():
        raise ValueError("External-case IDs must be unique")

    canonical: list[str] = []
    for row_index, value in frame["smiles"].items():
        try:
            canonical.append(canonicalize_smiles(value))
        except FeatureRepresentationError as error:
            raise ValueError(f"Invalid external SMILES at row {row_index}: {value}") from error
    frame["canonical_smiles"] = canonical
    if frame["canonical_smiles"].duplicated().any():
        raise ValueError("External cases contain duplicate canonical molecules")
    return frame


def audit_training_overlap(
    source: pd.DataFrame,
    *,
    task_root: Path | str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Audit exact canonical overlap against the complete 982-molecule corpus."""

    corpus = load_full_corpus(task_root)
    training = corpus.frame.copy()
    training["canonical_smiles"] = training["smiles"].map(canonicalize_smiles)
    if training["canonical_smiles"].duplicated().any():
        raise ValueError("The formal 982-molecule corpus is not canonical-molecule unique")
    lookup = training.set_index("canonical_smiles")[["sample_id", "label"]]

    audited = source.copy()
    audited["training_overlap"] = audited["canonical_smiles"].isin(lookup.index)
    audited["training_sample_id"] = audited["canonical_smiles"].map(lookup["sample_id"])
    audited["training_label"] = audited["canonical_smiles"].map(lookup["label"]).astype("Int64")
    conflict = audited["training_overlap"] & audited["training_label"].ne(audited["label"])
    if conflict.any():
        ids = audited.loc[conflict, "id"].tolist()
        raise ValueError(f"External/training label conflicts detected: {ids}")
    observed = set(audited.loc[audited["training_overlap"], "id"])
    if observed != EXPECTED_OVERLAP_IDS:
        raise ValueError(
            f"External overlap contract changed: expected {sorted(EXPECTED_OVERLAP_IDS)}, "
            f"found {sorted(observed)}"
        )
    return audited, corpus.input_files


def score_external_cases(
    audited: pd.DataFrame,
    *,
    runtime: ConsensusRuntime,
) -> pd.DataFrame:
    """Apply the fixed Workflow 60 ensemble without explanations."""

    payloads = runtime.predict_requests(
        {"smiles": smiles, "explain": False} for smiles in audited["smiles"]
    )
    if len(payloads) != len(audited):
        raise ValueError("Screening runtime did not preserve external-case row alignment")
    predictions = pd.DataFrame(payloads)
    if not predictions["status"].eq("ok").all():
        failed = predictions.loc[~predictions["status"].eq("ok"), "input_smiles"].tolist()
        raise ValueError(f"External inference failed for valid source molecules: {failed}")
    if predictions["canonical_smiles"].tolist() != audited["canonical_smiles"].tolist():
        raise ValueError("Screening runtime canonicalization differs from the overlap audit")

    scored = audited.reset_index(drop=True).copy()
    prediction_columns = [
        "rf_antagonistic_score",
        "rf_pred_label",
        "tabpfn_antagonistic_score",
        "tabpfn_pred_label",
        "lightgbm_antagonistic_score",
        "lightgbm_pred_label",
        "consensus_vote_count",
        "consensus_mean_score",
        "consensus_min_score",
        "consensus_max_score",
        "model_score_range",
        "consensus_all_antagonistic",
    ]
    for column in prediction_columns:
        scored[column] = predictions[column].to_numpy()
    scored = scored.rename(
        columns={"consensus_all_antagonistic": "strict_consensus_pred_label"}
    )
    scored["strict_consensus_case"] = [
        _case(int(label), int(prediction))
        for label, prediction in zip(
            scored["label"], scored["strict_consensus_pred_label"], strict=True
        )
    ]
    return scored


def summarize_accuracy(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize confusion counts, accuracy, and balanced accuracy."""

    cohorts = (
        ("external_20_all", scored),
        ("external_17_nonoverlap", scored.loc[~scored["training_overlap"]]),
    )
    rows: list[dict[str, Any]] = []
    for cohort_name, cohort in cohorts:
        labels = cohort["label"].astype(int)
        for predictor_name, prediction_column in PREDICTORS:
            predictions = cohort[prediction_column].astype(int)
            tp = int(((labels == 1) & (predictions == 1)).sum())
            fp = int(((labels == 0) & (predictions == 1)).sum())
            tn = int(((labels == 0) & (predictions == 0)).sum())
            fn = int(((labels == 1) & (predictions == 0)).sum())
            correct = tp + tn
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            rows.append(
                {
                    "cohort": cohort_name,
                    "predictor": predictor_name,
                    "n": int(len(cohort)),
                    "n_antagonistic": int((labels == 1).sum()),
                    "n_non_antagonistic": int((labels == 0).sum()),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "correct": correct,
                    "accuracy": correct / len(cohort),
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "balanced_accuracy": (sensitivity + specificity) / 2,
                }
            )
    return pd.DataFrame(rows)


def write_external_validation_package(
    *,
    repo_root: Path | str,
    task_root: Path | str,
    source_path: Path | str,
    destination: Path | str,
    runtime: ConsensusRuntime | None = None,
) -> ExternalValidationBundle:
    """Run the fixed external-case audit and write a traceable result package."""

    repo = Path(repo_root).resolve()
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    source_file = Path(source_path).resolve()
    source = load_external_source(source_file)
    audited, corpus_inputs = audit_training_overlap(source, task_root=task_root)
    active_runtime = runtime or ConsensusRuntime(WebArtifactPaths.from_repo(repo))
    scored = score_external_cases(audited, runtime=active_runtime)
    metrics = summarize_accuracy(scored)

    predictions_path = output / "external_20_predictions.csv"
    nonoverlap_path = output / "external_17_nonoverlap_predictions.csv"
    overlap_path = output / "overlap_audit.csv"
    metrics_path = output / "accuracy_summary.csv"
    report_path = output / "report.md"
    html_path = output / "report.html"
    manifest_path = output / "run_manifest.json"

    scored.to_csv(predictions_path, index=False)
    scored.loc[~scored["training_overlap"]].to_csv(nonoverlap_path, index=False)
    scored.loc[
        scored["training_overlap"],
        [
            "id",
            "name",
            "label",
            "canonical_smiles",
            "training_sample_id",
            "training_label",
        ],
    ].to_csv(overlap_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    report = _render_markdown_report(scored, metrics)
    report_path.write_text(report, encoding="utf-8")
    html_path.write_text(_render_html_report(scored, metrics), encoding="utf-8")

    screening_root = repo / "workflow" / "60_virtual_screen" / "results"
    rf_root = repo / "workflow" / "50_explain_rf" / "results"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "exploratory_literature_case_evaluation",
        "independent_evaluation": False,
        "interpretation": (
            "Small purposively balanced literature case set; three exact training overlaps; "
            "not a representative prospective prevalence sample."
        ),
        "label_contract": {"0": "non-antagonistic", "1": "antagonistic"},
        "source": {
            "path": str(source_file.relative_to(repo)),
            "sha256": _sha256(source_file),
            "rows": len(source),
            "class_counts": {"0": 10, "1": 10},
        },
        "overlap_audit": {
            "key": "stereochemistry-preserving RDKit canonical SMILES",
            "training_rows": 982,
            "overlap_count": int(scored["training_overlap"].sum()),
            "overlap_ids": scored.loc[scored["training_overlap"], "id"].tolist(),
            "nonoverlap_count": int((~scored["training_overlap"]).sum()),
        },
        "ensemble": {
            "models": [
                "rf__ecfp_2048",
                "tabpfn__ecfp_2048",
                "lgbm__ecfp_1024_rdkit2d_normalized_200",
            ],
            "selection_scope": "test_informed_screening_ensemble",
            "strict_consensus_rule": "all three antagonistic scores >= 0.5",
            "scores_calibrated": False,
            "inference_batch_rows": len(scored),
        },
        "metric_policy": {
            "reported_metrics": ["accuracy", "balanced_accuracy"],
            "confusion_counts": True,
            "auroc_reported": False,
            "auprc_reported": False,
        },
        "upstream_sha256": {
            "workflow_50_model_manifest": _sha256(rf_root / "model" / "model_manifest.json"),
            "workflow_60_run_manifest": _sha256(screening_root / "run_manifest.json"),
            "workflow_60_model_manifest": _sha256(
                screening_root / "models" / "model_manifest.json"
            ),
        },
        "training_input_sha256": corpus_inputs,
        "runtime": _runtime_snapshot(),
        "git": _git_snapshot(repo),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_checksums(output)
    return ExternalValidationBundle(
        root=output,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )


def _case(label: int, prediction: int) -> str:
    if label == 1:
        return "TP" if prediction == 1 else "FN"
    return "FP" if prediction == 1 else "TN"


def _render_markdown_report(scored: pd.DataFrame, metrics: pd.DataFrame) -> str:
    overlaps = ", ".join(
        f"{row.name} ({row.id})"
        for row in scored.loc[scored["training_overlap"], ["id", "name"]].itertuples()
    )
    strict = metrics.loc[metrics["predictor"].eq("Strict three-model consensus")]
    full = strict.loc[strict["cohort"].eq("external_20_all")].iloc[0]
    clean = strict.loc[strict["cohort"].eq("external_17_nonoverlap")].iloc[0]
    return f"""# Workflow 70 external-case evaluation

The fixed source contains 20 literature-derived μOR ligands (10 antagonistic and 10
non-antagonistic). Exact canonical-molecule auditing identified three training overlaps:
{overlaps}. The non-overlapping cohort therefore contains 17 cases (10 antagonistic and 7
non-antagonistic).

The strict LightGBM–TabPFN–RF consensus correctly classified **{int(full.correct)}/20**
(accuracy **{full.accuracy:.3f}**; balanced accuracy **{full.balanced_accuracy:.3f}**) before
overlap exclusion and **{int(clean.correct)}/17** (accuracy **{clean.accuracy:.3f}**; balanced
accuracy **{clean.balanced_accuracy:.3f}**) after exclusion.

## Classification metrics and confusion counts

{metrics.to_markdown(index=False, floatfmt=".3f")}

## Interpretation boundary

This is a small, purposively balanced literature case set rather than a random prospective sample.
The three overlapping non-antagonistic cases are disclosed, and the 17-case cohort is reported
separately. Accuracy and raw confusion counts are reported; AUROC and AUPRC are intentionally
omitted. Balanced accuracy is also reported as the unweighted mean of sensitivity and specificity.
The ensemble is the disclosed test-informed Workflow 60 screening artifact, and its scores are
uncalibrated.
"""


def _render_html_report(scored: pd.DataFrame, metrics: pd.DataFrame) -> str:
    overlaps = ", ".join(
        f"{html.escape(row.name)} ({html.escape(row.id)})"
        for row in scored.loc[scored["training_overlap"], ["id", "name"]].itertuples()
    )
    strict = metrics.loc[metrics["predictor"].eq("Strict three-model consensus")]
    full = strict.loc[strict["cohort"].eq("external_20_all")].iloc[0]
    clean = strict.loc[strict["cohort"].eq("external_17_nonoverlap")].iloc[0]
    metrics_table = metrics.to_html(
        index=False,
        float_format=lambda value: f"{value:.3f}",
        classes="metrics-table",
    )
    compounds_table = _compound_display_table(scored).to_html(
        index=False,
        escape=True,
        classes="compound-table",
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Workflow 70 external-case evaluation</title>
<style>
body{{max-width:1380px;margin:36px auto;padding:0 24px;font:16px/1.55 Arial,sans-serif;
color:#1f2933}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #d6d9dc;padding:7px 9px}}th{{background:#eef1f3;position:sticky;top:0}}
.metrics-table th,.metrics-table td{{text-align:right}}
.metrics-table th:nth-child(-n+2),.metrics-table td:nth-child(-n+2){{text-align:left}}
.table-wrap{{width:100%;overflow-x:auto;border:1px solid #d6d9dc}}
.compound-table{{min-width:1160px}}
.compound-table th,.compound-table td{{text-align:left;vertical-align:top}}
.compound-table td:nth-child(3){{min-width:150px;overflow-wrap:anywhere}}
.compound-table td:nth-child(n+6){{white-space:nowrap;text-align:center}}
.note{{background:#f5f1e8;border-left:4px solid #9c442d;padding:12px 16px}}
.legend{{color:#52606d;font-size:14px}}
</style></head><body>
<h1>Workflow 70 external-case evaluation</h1>
<p>The fixed source contains 20 literature-derived μOR ligands (10 antagonistic and 10
non-antagonistic). Exact canonical-molecule auditing identified three training overlaps:
{overlaps}. The non-overlapping cohort contains 17 cases.</p>
<p>The strict LightGBM–TabPFN–RF consensus correctly classified
<strong>{int(full.correct)}/20</strong> (accuracy <strong>{full.accuracy:.3f}</strong>; balanced
accuracy <strong>{full.balanced_accuracy:.3f}</strong>) before overlap exclusion and
<strong>{int(clean.correct)}/17</strong> (accuracy <strong>{clean.accuracy:.3f}</strong>; balanced
accuracy <strong>{clean.balanced_accuracy:.3f}</strong>) after exclusion.</p>
<h2>Classification metrics and confusion counts</h2>{metrics_table}
<p class="legend">Balanced accuracy is the unweighted mean of sensitivity and specificity.
Labels: 1 = antagonistic; 0 = non-antagonistic.</p>
<h2>Per-compound predictions</h2>
<p>The score/label columns show each model's uncalibrated antagonistic score followed by its
thresholded label. “Training overlap” denotes exact stereochemistry-preserving canonical-SMILES
overlap with the 982-molecule benchmark corpus. SMILES are retained in the downloadable prediction
CSVs but hidden from this compact view.</p>
<div class="table-wrap">{compounds_table}</div>
<p class="note">This small purposively balanced literature case set is exploratory. Accuracy,
balanced accuracy, and raw confusion counts are reported; AUROC and AUPRC are intentionally
omitted. Workflow 60 scores are uncalibrated, and the final ensemble is test-informed.</p>
</body></html>
"""


def _compound_display_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Format the complete molecule-level evidence for the standalone HTML report."""

    display = pd.DataFrame(
        {
            "ID": scored["id"],
            "Compound": scored["name"],
            "DOI": scored["doi"],
            "True class": scored["label"].map(
                {0: "0 · non-antagonistic", 1: "1 · antagonistic"}
            ),
            "Training overlap": scored["training_overlap"].map({False: "No", True: "Yes"}),
            "RF score / label": _score_label_column(
                scored, "rf_antagonistic_score", "rf_pred_label"
            ),
            "TabPFN score / label": _score_label_column(
                scored, "tabpfn_antagonistic_score", "tabpfn_pred_label"
            ),
            "LightGBM score / label": _score_label_column(
                scored, "lightgbm_antagonistic_score", "lightgbm_pred_label"
            ),
            "Votes": scored["consensus_vote_count"].map(lambda value: f"{int(value)}/3"),
            "Consensus label": scored["strict_consensus_pred_label"].map(
                {0: "0 · non-antagonistic", 1: "1 · antagonistic"}
            ),
            "Outcome": scored["strict_consensus_case"],
        }
    )
    return display


def _score_label_column(
    scored: pd.DataFrame,
    score_column: str,
    label_column: str,
) -> pd.Series:
    return pd.Series(
        (
            f"{float(score):.3f} / {int(label)}"
            for score, label in zip(
                scored[score_column],
                scored[label_column],
                strict=True,
            )
        ),
        index=scored.index,
    )


def _runtime_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("joblib", "numpy", "pandas", "rdkit", "scikit-learn", "tabpfn", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        cuda = {"available": False, "visible_devices": None, "device_name": None}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def _git_snapshot(repo: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"sha": revision, "dirty": dirty}


def _write_checksums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{_sha256(path)}  {path.relative_to(root)}" for path in paths]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ExternalValidationBundle",
    "audit_training_overlap",
    "load_external_source",
    "score_external_cases",
    "summarize_accuracy",
    "write_external_validation_package",
]
