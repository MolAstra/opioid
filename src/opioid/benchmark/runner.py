"""Leakage-safe reviewer benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from opioid.benchmark.artifacts import write_run_manifest
from opioid.benchmark.catalog import CandidateSpec, traditional_candidates
from opioid.benchmark.chemprop import (
    ChempropRunner,
    ChempropSpec,
    chemprop_candidates,
    chemprop_runtime_manifest,
)
from opioid.benchmark.context import LABEL_COLUMN, SMILES_COLUMN, RunContext
from opioid.benchmark.data import FoldSplit, load_fold_splits, load_test_split
from opioid.benchmark.estimators import build_estimator
from opioid.benchmark.metrics import compute_metrics
from opioid.benchmark.provenance import (
    development_split_snapshot,
    environment_snapshot,
    split_snapshot,
)
from opioid.benchmark.reporting import (
    merge_matrix_test,
    summarize_cv,
    write_tables,
)
from opioid.benchmark.status import StatusStore
from opioid.chemistry.representations import FeatureDataset, build_feature_dataset


ProgressCallback = Callable[[str], None]
TraditionalOrChemprop = CandidateSpec | ChempropSpec


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Paths for one complete protocol-level model-comparison run."""

    context: RunContext
    fold_metrics_csv: Path
    summary_csv: Path
    matrix_test_csv: Path
    run_manifest_json: Path


def run_protocol_benchmark(
    context: RunContext,
    *,
    suites: Iterable[str] = ("traditional", "chemprop"),
    evaluate_test: bool = True,
    progress: ProgressCallback | None = None,
) -> BenchmarkRunResult:
    """Evaluate the frozen candidate matrix for one persisted protocol.

    The independent test CSV is deliberately loaded only after validation has
    completed for every requested candidate. Every candidate's five fold models
    then predict the shared DPBench test set. No candidate is selected, ranked
    for deployment, or retrained on all development data.
    """

    selected_suites = _normalize_suites(suites)
    context.ensure_output_dirs()
    status = StatusStore(context.status_json)
    traditional = traditional_candidates() if "traditional" in selected_suites else ()
    chemprop = chemprop_candidates() if "chemprop" in selected_suites else ()
    candidates: tuple[TraditionalOrChemprop, ...] = (*traditional, *chemprop)
    if not candidates:
        raise ValueError("At least one benchmark suite must be selected")

    if progress:
        progress(f"[{context.protocol}] CV candidates: {len(candidates)}")
    development_snapshot = development_split_snapshot(context)
    folds = load_fold_splits(context)
    fold_rows = _evaluate_cv(
        context,
        folds,
        traditional,
        chemprop,
        status,
        development_snapshot,
        progress,
    )
    fold_metrics = pd.DataFrame(fold_rows)
    cv_summary = summarize_cv(fold_metrics)

    matrix_test: pd.DataFrame | None = None
    if evaluate_test:
        matrix_test = _evaluate_matrix_test(
            context,
            candidates,
            folds,
            status,
            development_snapshot,
            progress,
        )
        cv_summary = merge_matrix_test(cv_summary, matrix_test)
    table_paths = write_tables(context.output_dir, fold_metrics, cv_summary, matrix_test)
    manifest_path = write_run_manifest(
        context,
        {
            "suites": list(selected_suites),
            "candidate_count": len(candidates),
            "candidates": [_candidate_manifest(candidate) for candidate in candidates],
            "chemprop_runtime": chemprop_runtime_manifest() if chemprop else None,
            "environment": environment_snapshot(),
            # A CV-only run must not inspect test.csv even for provenance.
            # Its manifest therefore records only persisted train/validation
            # inputs; a matrix-test run records the complete split snapshot.
            "split_snapshot": (
                split_snapshot(context) if evaluate_test else development_split_snapshot(context)
            ),
            "test_snapshot_included": evaluate_test,
            "run_stage": "matrix_test" if evaluate_test else "cv_only",
            "candidate_matrix_frozen_before_test": evaluate_test,
            "test_policy": (
                "after all requested candidate CV jobs completed, evaluate every frozen candidate "
                "with five fold models on the shared test; do not use test metrics for selection, "
                "tuning, or threshold changes"
                if evaluate_test
                else "CV-only run: independent test CSV was not loaded"
            ),
        },
    )
    if progress:
        progress(f"[{context.protocol}] complete candidate matrix: {len(candidates)}")
        progress(f"[{context.protocol}] summary: {table_paths['summary']}")
    return BenchmarkRunResult(
        context=context,
        fold_metrics_csv=table_paths["fold_metrics"],
        summary_csv=table_paths["summary"],
        matrix_test_csv=table_paths["matrix_test"],
        run_manifest_json=manifest_path,
    )


def run_all_protocol_benchmarks(
    *,
    protocols: Iterable[str],
    suites: Iterable[str] = ("traditional", "chemprop"),
    evaluate_test: bool = True,
    split_root: Path | str | None = None,
    output_root: Path | str | None = None,
    progress: ProgressCallback | None = None,
) -> list[BenchmarkRunResult]:
    """Run isolated reviewer benchmarks for the requested named protocols."""

    return [
        run_protocol_benchmark(
            RunContext.for_protocol(
                protocol,
                **({"split_root": split_root} if split_root is not None else {}),
                **({"output_root": output_root} if output_root is not None else {}),
            ),
            suites=suites,
            evaluate_test=evaluate_test,
            progress=progress,
        )
        for protocol in protocols
    ]


def _evaluate_cv(
    context: RunContext,
    folds: list[FoldSplit],
    traditional: tuple[CandidateSpec, ...],
    chemprop: tuple[ChempropSpec, ...],
    status: StatusStore,
    development_snapshot: dict[str, object],
    progress: ProgressCallback | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chemprop_runner = ChempropRunner(seed=context.seed) if chemprop else None
    for fold in folds:
        if progress:
            progress(f"[{context.protocol}] fold {fold.fold_index}")
        for candidate in traditional:
            rows.append(
                _evaluate_traditional_fold(
                    context,
                    fold,
                    candidate,
                    status,
                    development_snapshot,
                    progress,
                )
            )
        for candidate in chemprop:
            if chemprop_runner is None:  # Defensive for type checkers and future edits.
                raise RuntimeError("Chemprop runner was not initialized")
            rows.append(
                _evaluate_chemprop_fold(
                    context,
                    fold,
                    candidate,
                    chemprop_runner,
                    status,
                    development_snapshot,
                    progress,
                )
            )
    return rows


def _evaluate_traditional_fold(
    context: RunContext,
    fold: FoldSplit,
    candidate: CandidateSpec,
    status: StatusStore,
    development_snapshot: dict[str, object],
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    job_id = _job_id("cv", context.protocol, candidate.candidate_id, fold.fold_index)
    metadata = _candidate_row_metadata(context.protocol, candidate)
    input_fingerprint = _job_input_fingerprint(
        context,
        candidate,
        stage="cv",
        fold_index=fold.fold_index,
        split_input_snapshot=development_snapshot,
    )
    completed = status.completed(job_id, input_fingerprint=input_fingerprint)
    if completed is not None:
        if progress:
            progress(f"  - {candidate.candidate_id} [resume]")
        return {"fold": fold.fold_index, **metadata, **dict(completed["metrics"])}

    if progress:
        progress(f"  - {candidate.candidate_id}")
    status.start(
        job_id,
        stage="cv",
        fold=fold.fold_index,
        input_fingerprint=input_fingerprint,
        **metadata,
    )
    try:
        train_data = _valid_feature_dataset(fold.train_df, candidate)
        valid_data = _valid_feature_dataset(fold.valid_df, candidate)
        model = build_estimator(candidate, seed=context.seed)
        model.fit(train_data.features, _labels(train_data))
        probabilities = _predict_probabilities(model, valid_data.features)
        metrics = {
            f"valid_{key}": value
            for key, value in compute_metrics(
                _labels(valid_data), probabilities, context.threshold
            ).items()
        }
        status.complete(
            job_id,
            metrics,
            stage="cv",
            fold=fold.fold_index,
            input_fingerprint=input_fingerprint,
            **metadata,
        )
        return {"fold": fold.fold_index, **metadata, **metrics}
    except KeyboardInterrupt:
        status.fail(
            job_id,
            "interrupted",
            stage="cv",
            fold=fold.fold_index,
            input_fingerprint=input_fingerprint,
            **metadata,
        )
        raise
    except Exception as error:
        status.fail(
            job_id,
            str(error),
            stage="cv",
            fold=fold.fold_index,
            input_fingerprint=input_fingerprint,
            **metadata,
        )
        raise


def _evaluate_chemprop_fold(
    context: RunContext,
    fold: FoldSplit,
    candidate: ChempropSpec,
    chemprop_runner: ChempropRunner,
    status: StatusStore,
    development_snapshot: dict[str, object],
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    job_id = _job_id("cv", context.protocol, candidate.candidate_id, fold.fold_index)
    metadata = _candidate_row_metadata(context.protocol, candidate)
    input_fingerprint = _job_input_fingerprint(
        context,
        candidate,
        stage="cv",
        fold_index=fold.fold_index,
        split_input_snapshot=development_snapshot,
    )
    completed = status.completed(job_id, input_fingerprint=input_fingerprint)
    if completed is not None:
        if progress:
            progress(f"  - {candidate.candidate_id} [resume]")
        return {"fold": fold.fold_index, **metadata, **dict(completed["metrics"])}

    if progress:
        progress(f"  - {candidate.candidate_id}")
    status.start(
        job_id,
        stage="cv",
        fold=fold.fold_index,
        input_fingerprint=input_fingerprint,
        **metadata,
    )
    try:
        checkpoints = chemprop_runner.fit_fold(
            candidate,
            fold.train_df,
            fold.valid_df,
            context.chemprop_dir
            / "cv"
            / chemprop_runner.output_namespace
            / f"fold_{fold.fold_index}",
        )
        probabilities = chemprop_runner.predict(
            candidate,
            checkpoints,
            fold.valid_df,
            context.chemprop_dir
            / "cv"
            / chemprop_runner.output_namespace
            / f"fold_{fold.fold_index}"
            / candidate.candidate_id
            / "valid_predictions.csv",
        )
        metrics = {
            f"valid_{key}": value
            for key, value in compute_metrics(
                fold.valid_df[LABEL_COLUMN].to_numpy(dtype=int), probabilities, context.threshold
            ).items()
        }
        status.complete(
            job_id,
            metrics,
            stage="cv",
            fold=fold.fold_index,
            input_fingerprint=input_fingerprint,
            **metadata,
        )
        return {"fold": fold.fold_index, **metadata, **metrics}
    except KeyboardInterrupt:
        status.fail(
            job_id,
            "interrupted",
            stage="cv",
            fold=fold.fold_index,
            input_fingerprint=input_fingerprint,
            **metadata,
        )
        raise
    except Exception as error:
        status.fail(
            job_id,
            str(error),
            stage="cv",
            fold=fold.fold_index,
            input_fingerprint=input_fingerprint,
            **metadata,
        )
        raise


def _evaluate_matrix_test(
    context: RunContext,
    candidates: tuple[TraditionalOrChemprop, ...],
    folds: list[FoldSplit],
    status: StatusStore,
    development_snapshot: dict[str, object],
    progress: ProgressCallback | None,
) -> pd.DataFrame:
    """Score the shared test set with every frozen candidate's five fold models."""

    if progress:
        progress(f"[{context.protocol}] matrix shared test: {len(candidates)} candidates")
    # This is intentionally the first point in the runner where the test table
    # is read. CV helpers above accept only train/validation frames.
    test_df = load_test_split(context)
    complete_snapshot = split_snapshot(context)
    rows: list[dict[str, Any]] = []
    chemprop_runner = ChempropRunner(seed=context.seed)
    for candidate in candidates:
        metadata = _candidate_row_metadata(context.protocol, candidate)
        if progress:
            progress(f"  - shared test: {_candidate_id(candidate)}")
        for fold in folds:
            job_id = _job_id(
                "matrix_test", context.protocol, _candidate_id(candidate), fold.fold_index
            )
            input_fingerprint = _job_input_fingerprint(
                context,
                candidate,
                stage="matrix_test",
                fold_index=fold.fold_index,
                split_input_snapshot={
                    "development": development_snapshot,
                    "complete": complete_snapshot,
                },
            )
            completed = status.completed(job_id, input_fingerprint=input_fingerprint)
            if completed is not None:
                rows.append({"fold": fold.fold_index, **metadata, **dict(completed["metrics"])})
                continue
            status.start(
                job_id,
                stage="matrix_test",
                fold=fold.fold_index,
                input_fingerprint=input_fingerprint,
                **metadata,
            )
            try:
                if isinstance(candidate, CandidateSpec):
                    train_data = _valid_feature_dataset(fold.train_df, candidate)
                    test_data = _valid_feature_dataset(test_df, candidate)
                    model = build_estimator(candidate, seed=context.seed)
                    model.fit(train_data.features, _labels(train_data))
                    probabilities = _predict_probabilities(model, test_data.features)
                    labels = _labels(test_data)
                    prediction_rows = test_data.rows
                else:
                    checkpoints = _chemprop_fold_checkpoints(
                        context, candidate, fold.fold_index, chemprop_runner
                    )
                    probabilities = chemprop_runner.predict(
                        candidate,
                        checkpoints,
                        test_df,
                        context.chemprop_dir
                        / "cv"
                        / chemprop_runner.output_namespace
                        / f"fold_{fold.fold_index}"
                        / candidate.candidate_id
                        / "test_predictions.csv",
                    )
                    labels = test_df[LABEL_COLUMN].to_numpy(dtype=int)
                    prediction_rows = test_df
                _write_test_predictions(
                    context,
                    f"{candidate.candidate_id}__fold_{fold.fold_index}",
                    prediction_rows,
                    probabilities,
                )
                metrics = compute_metrics(labels, probabilities, context.threshold)
                status.complete(
                    job_id,
                    metrics,
                    stage="matrix_test",
                    fold=fold.fold_index,
                    input_fingerprint=input_fingerprint,
                    **metadata,
                )
                rows.append({"fold": fold.fold_index, **metadata, **metrics})
            except KeyboardInterrupt:
                status.fail(
                    job_id,
                    "interrupted",
                    stage="matrix_test",
                    fold=fold.fold_index,
                    input_fingerprint=input_fingerprint,
                    **metadata,
                )
                raise
            except Exception as error:
                status.fail(
                    job_id,
                    str(error),
                    stage="matrix_test",
                    fold=fold.fold_index,
                    input_fingerprint=input_fingerprint,
                    **metadata,
                )
                raise
    return pd.DataFrame(rows)


def _chemprop_fold_checkpoints(
    context: RunContext,
    candidate: ChempropSpec,
    fold_index: int,
    chemprop_runner: ChempropRunner,
) -> list[Path]:
    root = (
        context.chemprop_dir
        / "cv"
        / chemprop_runner.output_namespace
        / f"fold_{fold_index}"
        / candidate.candidate_id
    )
    checkpoints = sorted(root.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"Missing Chemprop fold checkpoint for {candidate.candidate_id}, fold {fold_index}: {root}"
        )
    return checkpoints


def _valid_feature_dataset(df: pd.DataFrame, candidate: CandidateSpec) -> FeatureDataset:
    dataset = build_feature_dataset(df, candidate.feature_spec)
    if dataset.invalid_smiles:
        examples = ", ".join(repr(smiles) for smiles in dataset.invalid_smiles[:5])
        raise ValueError(
            f"Persisted benchmark split contains invalid SMILES for {candidate.candidate_id}: {examples}"
        )
    if dataset.n_rows != len(df):
        raise ValueError(f"Feature row count changed for {candidate.candidate_id}")
    return dataset


def _labels(dataset: FeatureDataset) -> np.ndarray:
    if dataset.labels is None:
        raise ValueError("Benchmark feature dataset is missing labels")
    return dataset.labels


def _predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Model predict_proba output must contain probabilities for both binary classes"
        )
    return probabilities[:, 1]


def _candidate_row_metadata(protocol: str, candidate: TraditionalOrChemprop) -> dict[str, str]:
    if isinstance(candidate, CandidateSpec):
        return {
            "protocol": protocol,
            "family": candidate.family,
            "algorithm": candidate.algorithm_id,
            "preset": candidate.algorithm_id,
            "representation": candidate.representation_id,
            "candidate_id": candidate.candidate_id,
        }
    return {
        "protocol": protocol,
        "family": candidate.family,
        "algorithm": candidate.algorithm,
        "preset": candidate.preset,
        "representation": candidate.representation,
        "candidate_id": candidate.candidate_id,
    }


def _candidate_manifest(candidate: TraditionalOrChemprop) -> dict[str, Any]:
    if isinstance(candidate, CandidateSpec):
        return candidate.manifest()
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "algorithm": candidate.algorithm,
        "preset": candidate.preset,
        "representation": candidate.representation,
        "molecule_featurizers": list(candidate.molecule_featurizers),
        # Runtime is part of the CV input fingerprint, so a CPU result cannot
        # be silently resumed after the GPU execution policy changes.
        "runtime": chemprop_runtime_manifest(),
    }


def _candidate_by_id(
    candidates: Iterable[TraditionalOrChemprop], candidate_id: str
) -> TraditionalOrChemprop:
    for candidate in candidates:
        if _candidate_id(candidate) == candidate_id:
            return candidate
    raise RuntimeError(f"Selected candidate is absent from the registered catalog: {candidate_id}")


def _candidate_id(candidate: TraditionalOrChemprop) -> str:
    return candidate.candidate_id


def _job_id(stage: str, protocol: str, candidate_id: str, fold_index: int | None = None) -> str:
    suffix = f".fold_{fold_index}" if fold_index is not None else ""
    return f"{stage}.{protocol}.{candidate_id}{suffix}"


def _job_input_fingerprint(
    context: RunContext,
    candidate: TraditionalOrChemprop,
    *,
    stage: str,
    fold_index: int | None,
    split_input_snapshot: dict[str, object],
) -> str:
    """Hash the exact persisted inputs and candidate specification for resume safety."""

    payload = {
        "protocol": context.protocol,
        "seed": context.seed,
        "threshold": context.threshold,
        "stage": stage,
        "fold_index": fold_index,
        "candidate": _candidate_manifest(candidate),
        "split_input_snapshot": split_input_snapshot,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_test_predictions(
    context: RunContext, candidate_id: str, rows: pd.DataFrame, probabilities: np.ndarray
) -> Path:
    output = rows[[SMILES_COLUMN, LABEL_COLUMN]].copy().reset_index(drop=True)
    if len(output) != len(probabilities):
        raise ValueError("Test prediction count does not match test rows")
    output["pred_prob"] = np.asarray(probabilities, dtype=float)
    output["pred_label"] = (output["pred_prob"] >= context.threshold).astype(int)
    path = context.test_predictions_dir / f"{candidate_id}.csv"
    output.to_csv(path, index=False)
    return path


def _normalize_suites(suites: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(item) for item in suites))
    unknown = set(selected).difference({"traditional", "chemprop"})
    if unknown:
        raise ValueError(f"Unknown benchmark suites: {sorted(unknown)}")
    return selected
