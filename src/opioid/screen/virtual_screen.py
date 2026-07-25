"""Full-corpus three-model consensus virtual screening."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opioid.benchmark.catalog import CandidateSpec, traditional_candidates
from opioid.benchmark.display_names import get_model_display
from opioid.benchmark.estimators import DEFAULT_SEED, build_estimator
from opioid.benchmark.family_selection import (
    rank_test_informed_families,
    select_validation_representatives,
)
from opioid.chemistry.representations import (
    FeatureRepresentationError,
    canonicalize_smiles,
    featurize_smiles,
    get_feature_spec,
)
from opioid.explain.full_rf import load_full_corpus, load_full_rf_bundle


SCHEMA_VERSION = 3
THRESHOLD = 0.5
RF_ID = "rf__ecfp_2048"
TABPFN_ID = "tabpfn__ecfp_2048"
LGBM_ID = "lgbm__ecfp_1024_rdkit2d_normalized_200"
ENSEMBLE_IDS = (RF_ID, TABPFN_ID, LGBM_ID)
COMBINED_FEATURE_ID = "ecfp_2048_rdkit2d_normalized_200"
SCORE_COLUMNS = {
    RF_ID: "rf_antagonistic_score",
    TABPFN_ID: "tabpfn_antagonistic_score",
    LGBM_ID: "lightgbm_antagonistic_score",
}
LABEL_COLUMNS = {
    RF_ID: "rf_pred_label",
    TABPFN_ID: "tabpfn_pred_label",
    LGBM_ID: "lightgbm_pred_label",
}


@dataclass(frozen=True)
class SourceSpec:
    """One workflow-owned screening source snapshot."""

    name: str
    filename: str
    smiles_column: str


SOURCE_SPECS = (
    SourceSpec("gpcrdb", "gpcrdb.csv", "smiles"),
    SourceSpec("zinc", "zinc_scaffold.csv", "smiles"),
    SourceSpec("reinvent", "reinvent_stage2_1.csv", "SMILES"),
    SourceSpec("ouroboros", "opioid_ouroboros.csv", "smiles"),
)


@dataclass(frozen=True)
class ScreeningBundle:
    """Main paths produced by a complete screening run."""

    root: Path
    report_path: Path
    manifest_path: Path
    consensus_path: Path
    summary_path: Path


def _candidate(candidate_id: str) -> CandidateSpec:
    return next(
        candidate
        for candidate in traditional_candidates()
        if candidate.candidate_id == candidate_id
    )


def extract_lgbm_features(combined: np.ndarray) -> np.ndarray:
    """Extract ECFP4-1024 + RDKit2D-200 from the shared 2248-column matrix."""

    if combined.ndim != 2 or combined.shape[1] != 2248:
        raise ValueError(
            "LightGBM feature routing requires ECFP4-2048 + RDKit2D-200 input"
        )
    return np.concatenate((combined[:, :1024], combined[:, 2048:]), axis=1)


def predict_lgbm_proba(model: Any, combined: np.ndarray) -> np.ndarray:
    """Predict with stable feature names while preserving the benchmark column order."""

    features = extract_lgbm_features(combined)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        features = pd.DataFrame(features, columns=list(feature_names))
    return model.predict_proba(features)[:, 1]


def clean_screening_sources(
    source_root: Path | str,
    training_smiles: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Canonicalize, audit and deduplicate sources before featurization.

    Deduplication is performed by stereochemistry-preserving canonical RDKit
    SMILES. The first row for a canonical molecule is retained within each
    source. A molecule shared by different sources is predicted once globally
    and remains represented in every source-specific output.
    """

    root = Path(source_root)
    training_canonical = {canonicalize_smiles(value) for value in training_smiles}
    audit_rows: list[dict[str, Any]] = []
    representatives: dict[str, pd.DataFrame] = {}

    for spec in SOURCE_SPECS:
        path = root / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing screening source snapshot: {path}")
        source = pd.read_csv(path)
        if spec.smiles_column not in source:
            raise ValueError(f"{path} is missing SMILES column {spec.smiles_column!r}")

        seen: dict[str, int] = {}
        representative_positions: list[int] = []
        canonical_by_position: dict[int, str] = {}
        duplicate_counts: defaultdict[str, int] = defaultdict(int)
        for position, value in enumerate(source[spec.smiles_column].tolist()):
            raw_smiles = "" if pd.isna(value) else str(value).strip()
            try:
                canonical = canonicalize_smiles(raw_smiles)
            except FeatureRepresentationError:
                audit_rows.append(
                    {
                        "source": spec.name,
                        "source_row": position,
                        "raw_smiles": raw_smiles,
                        "canonical_smiles": "",
                        "status": "invalid_smiles",
                        "representative_source_row": pd.NA,
                    }
                )
                continue

            if canonical in training_canonical:
                audit_rows.append(
                    {
                        "source": spec.name,
                        "source_row": position,
                        "raw_smiles": raw_smiles,
                        "canonical_smiles": canonical,
                        "status": "training_overlap",
                        "representative_source_row": pd.NA,
                    }
                )
                continue

            duplicate_counts[canonical] += 1
            if canonical in seen:
                status = "within_source_duplicate"
                representative = seen[canonical]
            else:
                status = "eligible"
                representative = position
                seen[canonical] = position
                representative_positions.append(position)
                canonical_by_position[position] = canonical
            audit_rows.append(
                {
                    "source": spec.name,
                    "source_row": position,
                    "raw_smiles": raw_smiles,
                    "canonical_smiles": canonical,
                    "status": status,
                    "representative_source_row": representative,
                }
            )

        kept = source.iloc[representative_positions].copy().reset_index(drop=True)
        kept.insert(0, "source_row", representative_positions)
        kept.insert(1, "source", spec.name)
        kept.insert(
            2,
            "canonical_smiles",
            [canonical_by_position[position] for position in representative_positions],
        )
        kept.insert(
            3,
            "source_duplicate_count",
            [duplicate_counts[value] for value in kept["canonical_smiles"]],
        )
        representatives[spec.name] = kept

    audit = pd.DataFrame(audit_rows)
    memberships: defaultdict[str, set[str]] = defaultdict(set)
    raw_counts: defaultdict[str, int] = defaultdict(int)
    for name, frame in representatives.items():
        for row in frame.itertuples(index=False):
            memberships[row.canonical_smiles].add(name)
            raw_counts[row.canonical_smiles] += int(row.source_duplicate_count)

    global_rows = [
        {
            "screen_id": _screen_id(canonical),
            "canonical_smiles": canonical,
            "sources": ";".join(sorted(names)),
            "source_count": len(names),
            "source_row_count": raw_counts[canonical],
        }
        for canonical, names in sorted(memberships.items())
    ]
    global_candidates = pd.DataFrame(global_rows)
    if not global_candidates["screen_id"].is_unique:
        raise RuntimeError("Screen identifiers collided; increase their digest length")
    return global_candidates, audit, representatives


def add_consensus_columns(frame: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Add fixed-threshold predictions and all-three consensus fields."""

    work = frame.copy()
    score_columns = [SCORE_COLUMNS[candidate_id] for candidate_id in ENSEMBLE_IDS]
    missing = sorted(set(score_columns).difference(work.columns))
    if missing:
        raise ValueError(f"Consensus scores are missing columns: {missing}")
    for candidate_id in ENSEMBLE_IDS:
        work[LABEL_COLUMNS[candidate_id]] = (
            work[SCORE_COLUMNS[candidate_id]].to_numpy(dtype=float) >= threshold
        ).astype(int)
    labels = [LABEL_COLUMNS[candidate_id] for candidate_id in ENSEMBLE_IDS]
    work["consensus_all_antagonistic"] = work[labels].eq(1).all(axis=1).astype(int)
    work["consensus_all_non_antagonistic"] = work[labels].eq(0).all(axis=1).astype(int)
    work["consensus_vote_count"] = work[labels].sum(axis=1).astype(int)
    work["consensus_mean_score"] = work[score_columns].mean(axis=1)
    work["consensus_min_score"] = work[score_columns].min(axis=1)
    work["consensus_max_score"] = work[score_columns].max(axis=1)
    work["model_score_range"] = work["consensus_max_score"] - work["consensus_min_score"]
    return work


def write_virtual_screen_package(
    *,
    task_root: Path | str,
    rf_bundle_root: Path | str,
    source_root: Path | str,
    benchmark_report_root: Path | str,
    destination: Path | str,
    batch_size: int = 1024,
    progress: Callable[[str], None] | None = None,
) -> ScreeningBundle:
    """Train the full-data ensemble, screen all snapshots and write a result bundle."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    log = progress or (lambda _message: None)
    output = Path(destination)
    model_dir = output / "models"
    prediction_dir = output / "predictions"
    figure_dir = output / "figures"
    for path in (output, model_dir, prediction_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)

    selection_evidence = _selection_evidence(Path(benchmark_report_root))
    _write_selection_table(selection_evidence, output)
    corpus = load_full_corpus(task_root)
    training = corpus.frame.copy()
    log(f"[data] full labeled corpus: {len(training)} rows")
    global_candidates, cleaning_audit, source_representatives = clean_screening_sources(
        source_root,
        training["smiles"].astype(str),
    )
    log(
        "[data] canonical non-training candidates: "
        f"{len(global_candidates)} global unique molecules"
    )

    combined_spec = get_feature_spec(COMBINED_FEATURE_ID)
    training_vectors = np.stack(
        [featurize_smiles(value, combined_spec) for value in training["smiles"].astype(str)],
        axis=0,
    ).astype(np.float32, copy=False)
    labels = training["label"].to_numpy(dtype=int)

    rf_model, rf_manifest = load_full_rf_bundle(rf_bundle_root)
    lgbm_candidate = _candidate(LGBM_ID)
    tabpfn_candidate = _candidate(TABPFN_ID)
    lgbm_model = build_estimator(lgbm_candidate, seed=DEFAULT_SEED)
    tabpfn_model = build_estimator(tabpfn_candidate, seed=DEFAULT_SEED)
    log("[train] fitting LightGBM / ECFP4-1024 + RDKit2D (200-d) on all 982 labels")
    lgbm_model.fit(extract_lgbm_features(training_vectors), labels)
    log("[train] fitting TabPFN / ECFP4-2048 on GPU")
    tabpfn_model.fit(training_vectors[:, :2048], labels)

    predictions: list[pd.DataFrame] = []
    feature_errors: list[dict[str, str]] = []
    total = len(global_candidates)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = global_candidates.iloc[start:stop]
        vectors: list[np.ndarray] = []
        kept_positions: list[int] = []
        for position, canonical in zip(batch.index, batch["canonical_smiles"], strict=True):
            try:
                vectors.append(featurize_smiles(str(canonical), combined_spec))
                kept_positions.append(int(position))
            except FeatureRepresentationError as error:
                feature_errors.append(
                    {
                        "screen_id": str(global_candidates.at[position, "screen_id"]),
                        "canonical_smiles": str(canonical),
                        "error": str(error),
                    }
                )
        if vectors:
            features = np.stack(vectors, axis=0).astype(np.float32, copy=False)
            predicted = global_candidates.loc[kept_positions].copy()
            predicted[SCORE_COLUMNS[RF_ID]] = rf_model.predict_proba(features[:, :2048])[:, 1]
            predicted[SCORE_COLUMNS[LGBM_ID]] = predict_lgbm_proba(lgbm_model, features)
            predicted[SCORE_COLUMNS[TABPFN_ID]] = tabpfn_model.predict_proba(
                features[:, :2048]
            )[:, 1]
            predictions.append(predicted)
        log(f"[screen] predicted {stop:,}/{total:,} canonical candidates")

    scored = add_consensus_columns(pd.concat(predictions, ignore_index=True))
    scored = scored.sort_values(
        ["consensus_all_antagonistic", "consensus_mean_score", "consensus_min_score", "screen_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    scored.insert(1, "global_rank", np.arange(1, len(scored) + 1, dtype=int))
    scored.to_csv(output / "all_screened_candidates.csv", index=False)
    consensus = scored.loc[scored["consensus_all_antagonistic"] == 1].copy()
    consensus.to_csv(output / "consensus_antagonistic_candidates.csv", index=False)

    feature_error_frame = pd.DataFrame(
        feature_errors,
        columns=["screen_id", "canonical_smiles", "error"],
    )
    feature_error_frame.to_csv(output / "feature_errors.csv", index=False)
    cleaning_audit.to_csv(output / "cleaning_audit.csv", index=False)

    source_outputs: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        representative = source_representatives[spec.name]
        source_output = representative.merge(scored, on="canonical_smiles", how="inner")
        source_output = source_output.sort_values(
            ["consensus_all_antagonistic", "consensus_mean_score", "consensus_min_score"],
            ascending=[False, False, False],
            kind="stable",
        ).reset_index(drop=True)
        source_output.insert(0, "source_rank", np.arange(1, len(source_output) + 1, dtype=int))
        source_output.to_csv(prediction_dir / f"{spec.name}.csv", index=False)
        source_outputs[spec.name] = source_output

        audit = cleaning_audit.loc[cleaning_audit["source"] == spec.name]
        all1 = int(source_output["consensus_all_antagonistic"].sum())
        all0 = int(source_output["consensus_all_non_antagonistic"].sum())
        summary_rows.append(
            {
                "source": spec.name,
                "n_source_rows": len(audit),
                "n_invalid_smiles": int((audit["status"] == "invalid_smiles").sum()),
                "n_training_overlap_rows": int((audit["status"] == "training_overlap").sum()),
                "n_within_source_duplicate_rows": int(
                    (audit["status"] == "within_source_duplicate").sum()
                ),
                "n_unique_eligible": int((audit["status"] == "eligible").sum()),
                "n_feature_errors": int(
                    representative["canonical_smiles"]
                    .isin(feature_error_frame.get("canonical_smiles", pd.Series(dtype=str)))
                    .sum()
                ),
                "n_screened": len(source_output),
                "n_consensus_antagonistic": all1,
                "consensus_antagonistic_rate": all1 / max(len(source_output), 1),
                "n_consensus_non_antagonistic": all0,
                "consensus_non_antagonistic_rate": all0 / max(len(source_output), 1),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "screening_summary.csv", index=False)

    _write_figures(summary, source_outputs, scored, figure_dir)
    model_paths = _write_model_bundle(
        model_dir=model_dir,
        lgbm_model=lgbm_model,
        tabpfn_model=tabpfn_model,
        rf_bundle_root=Path(rf_bundle_root),
        rf_manifest=rf_manifest,
    )
    manifest = _build_manifest(
        output=output,
        source_root=Path(source_root),
        corpus=corpus,
        rf_manifest=rf_manifest,
        model_paths=model_paths,
        selection_evidence=selection_evidence,
        batch_size=batch_size,
        summary=summary,
        global_count=len(scored),
        consensus_count=len(consensus),
    )
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = output / "report.md"
    report_path.write_text(
        _build_report(summary, len(scored), len(consensus), selection_evidence),
        encoding="utf-8",
    )
    _write_checksums(output)
    return ScreeningBundle(
        root=output,
        report_path=report_path,
        manifest_path=manifest_path,
        consensus_path=output / "consensus_antagonistic_candidates.csv",
        summary_path=output / "screening_summary.csv",
    )


def _write_model_bundle(
    *,
    model_dir: Path,
    lgbm_model: Any,
    tabpfn_model: Any,
    rf_bundle_root: Path,
    rf_manifest: dict[str, Any],
) -> dict[str, Path]:
    lgbm_path = model_dir / f"{LGBM_ID}.joblib"
    tabpfn_path = model_dir / f"{TABPFN_ID}.joblib"
    joblib.dump(lgbm_model, lgbm_path)
    joblib.dump(tabpfn_model, tabpfn_path)
    paths = {LGBM_ID: lgbm_path, TABPFN_ID: tabpfn_path}
    rf_path = rf_bundle_root / str(rf_manifest["artifact_path"])
    model_manifest = {
        "schema_version": SCHEMA_VERSION,
        "training_scope": "full_deduplicated_labeled_corpus",
        "training_rows": 982,
        "independent_evaluation": False,
        "selection_scope": "test_informed_screening_ensemble",
        "models": {
            RF_ID: {
                "role": "reused_upstream_artifact",
                "artifact_path": (
                    "workflow/50_explain_rf/results/model/rf__ecfp_2048.joblib"
                ),
                "artifact_sha256": _sha256(rf_path),
                "artifact_published": False,
            },
            LGBM_ID: {
                "role": "workflow_60_full_data_refit",
                "artifact_path": lgbm_path.name,
                "artifact_sha256": _sha256(lgbm_path),
                "artifact_published": False,
            },
            TABPFN_ID: {
                "role": "workflow_60_full_data_refit",
                "artifact_path": tabpfn_path.name,
                "artifact_sha256": _sha256(tabpfn_path),
                "artifact_published": False,
            },
        },
        "artifact_regeneration_command": (
            "./workflow/60_virtual_screen/run.sh all --replace --batch-size 8192"
        ),
    }
    (model_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _selection_evidence(report_root: Path) -> dict[str, Any]:
    path = report_root / "all_candidates.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Workflow 40 benchmark table: {path}")
    table = pd.read_csv(path)
    representatives = select_validation_representatives(table, protocol="datasail")
    ranked = rank_test_informed_families(representatives, top_n=3)
    selected_ids = set(
        ranked.loc[ranked["selected_for_screening"], "candidate_id"].astype(str)
    )
    if selected_ids != set(ENSEMBLE_IDS):
        raise ValueError(
            "DataSAIL two-stage selection no longer matches the implemented screening ensemble: "
            f"expected {sorted(ENSEMBLE_IDS)}, observed {sorted(selected_ids)}"
        )
    columns = [
        "test_roc_rank",
        "selected_for_screening",
        "model_family",
        "candidate_id",
        "model_name",
        "model_type",
        "valid_roc_mean",
        "valid_roc_std",
        "test_roc_mean",
        "test_roc_std",
        "test_prc_mean",
    ]
    return {
        "scope": "test_informed_screening_ensemble",
        "warning": (
            "Within each of eight model families, the candidate was selected by DataSAIL "
            "validation AUROC. The three families were then selected by shared-test AUROC, so "
            "those tests are not independent evaluation of the final screening ensemble."
        ),
        "protocol": "datasail",
        "family_selection": "maximize valid_roc_mean; candidate_id ascending tie-break",
        "model_selection": "rank validation-selected families by test_roc_mean; top 3",
        "model_families": ranked["model_family"].astype(str).tolist(),
        "selected_candidate_ids": ranked.loc[
            ranked["selected_for_screening"], "candidate_id"
        ].astype(str).tolist(),
        "source": str(path),
        "source_sha256": _sha256(path),
        "rows": ranked[columns].to_dict("records"),
    }


def _selection_table_frame(selection_evidence: dict[str, Any]) -> pd.DataFrame:
    """Format the auditable two-stage DataSAIL selection as a manuscript table."""

    frame = pd.DataFrame(selection_evidence["rows"]).copy()
    frame["Validation-best representation"] = frame["model_type"]
    frame["Validation AUROC"] = frame.apply(
        lambda row: f'{row["valid_roc_mean"]:.3f} ± {row["valid_roc_std"]:.3f}',
        axis=1,
    )
    frame["Shared-test AUROC"] = frame.apply(
        lambda row: f'{row["test_roc_mean"]:.3f} ± {row["test_roc_std"]:.3f}',
        axis=1,
    )
    frame["Screening ensemble"] = frame["selected_for_screening"].map(
        {True: "Top 3", False: "—"}
    )
    return frame[
        [
            "test_roc_rank",
            "model_family",
            "Validation-best representation",
            "candidate_id",
            "Validation AUROC",
            "Shared-test AUROC",
            "Screening ensemble",
        ]
    ].rename(
        columns={
            "test_roc_rank": "Rank",
            "model_family": "Model family",
            "candidate_id": "Candidate ID",
        }
    )


def _write_selection_table(
    selection_evidence: dict[str, Any],
    destination: Path,
) -> tuple[Path, Path]:
    """Write CSV and self-contained HTML forms of the two-stage selection table."""

    table = _selection_table_frame(selection_evidence)
    csv_path = destination / "datasail_two_stage_model_selection.csv"
    html_path = destination / "datasail_two_stage_model_selection.html"
    table.to_csv(csv_path, index=False)
    body = table.to_html(
        index=False,
        escape=True,
        classes="selection-table",
        border=0,
        justify="left",
    )
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DataSAIL two-stage model selection</title>
  <style>
    :root {{ color: #17232b; background: #f5f2eb; font-family: Arial, sans-serif; }}
    body {{ margin: 0; padding: 32px; }}
    main {{ max-width: 1220px; margin: auto; padding: 28px; background: white;
            border: 1px solid #d8d3c8; }}
    h1 {{ margin: 0 0 8px; font-size: 25px; font-weight: 500; }}
    .subtitle, .note {{ color: #5f6e77; line-height: 1.5; }}
    .table-wrap {{ margin: 22px 0 16px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th {{ padding: 11px 10px; background: #274f61; color: white; text-align: left;
          white-space: nowrap; font-weight: 600; }}
    td {{ padding: 10px; border-bottom: 1px solid #dedbd4; vertical-align: top; }}
    tbody tr:nth-child(-n+3) {{ background: #e9f4ed; }}
    tbody tr:nth-child(-n+3) td:first-child {{ color: #26633e; font-weight: 700; }}
    td:nth-child(1), td:nth-child(5), td:nth-child(6), td:nth-child(7) {{
      white-space: nowrap;
    }}
    code {{ font-size: 12px; }}
    .note {{ margin: 0; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>DataSAIL two-stage model selection</h1>
  <p class="subtitle">Within each of eight model families, validation mean AUROC fixes one
  representation/preset. Shared-test mean AUROC then ranks those frozen family representatives
  and identifies the three-model screening ensemble.</p>
  <div class="table-wrap">{body}</div>
  <p class="note"><strong>Values are mean ± sample SD across five folds.</strong>
  The highlighted Top 3 are LightGBM, TabPFN, and Random Forest. Because shared-test AUROC is
  used for family selection, the final ensemble is test-informed and these test results are not
  independent validation of that ensemble.</p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return csv_path, html_path


def _build_manifest(
    *,
    output: Path,
    source_root: Path,
    corpus: Any,
    rf_manifest: dict[str, Any],
    model_paths: dict[str, Path],
    selection_evidence: dict[str, Any],
    batch_size: int,
    summary: pd.DataFrame,
    global_count: int,
    consensus_count: int,
) -> dict[str, Any]:
    models = {}
    for candidate_id in ENSEMBLE_IDS:
        display = get_model_display(candidate_id)
        models[candidate_id] = {
            "model_name": display.model_name,
            "model_type": display.model_type,
            "score_column": SCORE_COLUMNS[candidate_id],
            "label_column": LABEL_COLUMNS[candidate_id],
            "artifact": (
                "workflow/50_explain_rf/results/model/rf__ecfp_2048.joblib"
                if candidate_id == RF_ID
                else str(model_paths[candidate_id].relative_to(output))
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "full_corpus_consensus_virtual_screen",
        "label_contract": {"0": "non-antagonistic", "1": "antagonistic"},
        "threshold": THRESHOLD,
        "consensus_rule": "all three model scores >= 0.5",
        "selection_scope": "test_informed_screening_ensemble",
        "independent_evaluation": False,
        "scores_calibrated": False,
        "training_scope": "all 982 deduplicated labeled molecules",
        "training_input_sha256": corpus.input_files,
        "rf_upstream_artifact_sha256": rf_manifest["artifact_sha256"],
        "models": models,
        "selection_evidence": selection_evidence,
        "cleaning": {
            "canonicalization": "RDKit canonical isomeric SMILES",
            "salt_or_charge_standardization": False,
            "training_overlap_key": "canonical_smiles",
            "within_source_deduplication_key": "canonical_smiles",
            "cross_source_prediction_deduplication_key": "canonical_smiles",
        },
        "sources": {
            spec.name: {
                "path": str(source_root / spec.filename),
                "sha256": _sha256(source_root / spec.filename),
                "smiles_column": spec.smiles_column,
            }
            for spec in SOURCE_SPECS
        },
        "batch_size": batch_size,
        "counts": {
            "global_screened": global_count,
            "global_consensus_antagonistic": consensus_count,
            "by_source": summary.to_dict("records"),
        },
        "runtime": _runtime_snapshot(),
    }


def _runtime_snapshot() -> dict[str, Any]:
    packages = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "rdkit",
        "descriptastorus",
        "lightgbm",
        "tabpfn",
        "torch",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def _write_figures(
    summary: pd.DataFrame,
    source_outputs: dict[str, pd.DataFrame],
    scored: pd.DataFrame,
    destination: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = 0.25
    ax.bar(x - width, summary["n_source_rows"], width, label="Source rows", color="#a9bac8")
    ax.bar(x, summary["n_screened"], width, label="Screened unique", color="#4f7c9b")
    ax.bar(
        x + width,
        summary["n_consensus_antagonistic"],
        width,
        label="All-three antagonistic",
        color="#c45d2f",
    )
    ax.set_xticks(x, summary["source"])
    ax.set_ylabel("Molecule count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "source_funnel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x, summary["consensus_antagonistic_rate"], color="#c45d2f")
    ax.set_xticks(x, summary["source"])
    ax.set_ylabel("All-three antagonistic consensus rate")
    ax.set_ylim(0, max(0.05, float(summary["consensus_antagonistic_rate"].max()) * 1.15))
    fig.tight_layout()
    fig.savefig(destination / "consensus_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_consensus_mean_score_ecdf(source_outputs, destination)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, candidate_id in zip(axes, ENSEMBLE_IDS, strict=True):
        column = SCORE_COLUMNS[candidate_id]
        for name, frame in source_outputs.items():
            ax.hist(
                frame[column],
                bins=np.linspace(0, 1, 41),
                density=True,
                histtype="step",
                linewidth=1.5,
                label=name,
            )
        display = get_model_display(candidate_id)
        ax.set_title(f"{display.model_name}\n{display.model_type}", fontsize=9)
        ax.set_xlabel("Antagonistic score")
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    scores = scored[[SCORE_COLUMNS[item] for item in ENSEMBLE_IDS]].corr().to_numpy()
    labels = [get_model_display(item).model_name for item in ENSEMBLE_IDS]
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(scores, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3), labels, rotation=30, ha="right")
    ax.set_yticks(range(3), labels)
    for row in range(3):
        for column in range(3):
            ax.text(column, row, f"{scores[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(destination / "model_score_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_consensus_mean_score_ecdf(
    source_outputs: dict[str, pd.DataFrame],
    destination: Path,
) -> Path:
    """Plot one empirical CDF of the three-model mean score per screening source."""

    colors = {
        "gpcrdb": "#0077B6",
        "zinc": "#F05A00",
        "reinvent": "#168B54",
        "ouroboros": "#7B4FA3",
    }
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for name, frame in source_outputs.items():
        scores = np.sort(frame["consensus_mean_score"].to_numpy(dtype=float))
        if not len(scores):
            continue
        cumulative = np.arange(1, len(scores) + 1, dtype=float) / len(scores)
        ax.step(
            np.concatenate(([0.0], scores)),
            np.concatenate(([0.0], cumulative)),
            where="post",
            linewidth=2.0,
            color=colors.get(name),
            label=f"{name.upper()} (n={len(scores):,})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Three-model mean antagonistic score")
    ax.set_ylabel("Cumulative fraction of screened molecules")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="both", color="#DCE3E8", linewidth=0.8, alpha=0.65)
    fig.tight_layout()
    output_path = destination / "consensus_mean_score_ecdf.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_report(
    summary: pd.DataFrame,
    global_count: int,
    consensus_count: int,
    selection_evidence: dict[str, Any],
) -> str:
    table = summary.to_markdown(index=False, floatfmt=".4f")
    rate = consensus_count / max(global_count, 1)
    return f"""# Workflow 60 virtual-screen report

The workflow screened **{global_count:,}** globally unique, canonical, non-training molecules.
The fixed 0.5 all-three rule retained **{consensus_count:,}** putative antagonistic candidates
({rate:.2%}). `label = 1` and every reported score mean antagonistic.

## Source summary

{table}

## Ensemble

- Random Forest / ECFP4 (2048-bit), reused from Workflow 50.
- TabPFN / ECFP4 (2048-bit), refit on all 982 labels.
- LightGBM / ECFP4 (1024-bit) + RDKit2D (200-d), refit on all 982 labels.
- Consensus requires every uncalibrated class-1 score to be at least 0.5.
- The complete eight-family selection table is available in
  `datasail_two_stage_model_selection.html` and `.csv`.

## Interpretation boundary

This is a prioritization workflow, not an independent performance evaluation. DataSAIL validation
AUROC selected one candidate in each of eight model families, including Chemprop; shared-test
AUROC then selected the three families. This final ensemble is marked
`{selection_evidence["scope"]}`, and its shared-test tables must not be presented as unused
independent validation. Scores are uncalibrated and require experimental follow-up.

Cleaning uses stereochemistry-preserving RDKit canonical SMILES for invalid-row detection,
within-source deduplication, cross-source prediction deduplication, and exclusion of all 982 labeled
training molecules. It does not neutralize charges or strip salts. Full row-level decisions are in
`cleaning_audit.csv`.
"""


def _write_checksums(output: Path) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and path.suffix != ".joblib"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(output)}" for path in paths]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _screen_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:20]
    return f"screen_{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ENSEMBLE_IDS",
    "LABEL_COLUMNS",
    "LGBM_ID",
    "RF_ID",
    "SCORE_COLUMNS",
    "SOURCE_SPECS",
    "ScreeningBundle",
    "SourceSpec",
    "add_consensus_columns",
    "clean_screening_sources",
    "extract_lgbm_features",
    "predict_lgbm_proba",
    "TABPFN_ID",
    "write_virtual_screen_package",
]
