from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from opioid.baselines import config
from opioid.baselines.artifacts import save_classic_model, write_rank_outputs, write_runtime_config
from opioid.baselines.data import load_development_set, load_fold_splits, load_test_split
from opioid.baselines.features import build_feature_dataset
from opioid.baselines.io import ensure_output_dirs, write_run_config, write_test_predictions
from opioid.baselines.metrics import compute_metrics
from opioid.baselines.models import (
    classic_model_names,
    fit_chemprop_variant,
    train_chemprop_variant,
    train_classic_and_predict,
)
from opioid.baselines.status import (
    get_job,
    is_completed,
    load_status,
    mark_completed,
    mark_failed,
    mark_started,
)


def _model_family(model_label: str) -> str:
    if model_label in {"svm", "rf", "xgb", "lgbm"}:
        return "simple_ml"
    if model_label == "tabpfn":
        return "tabpfn"
    if model_label.startswith("chemprop"):
        return "chemprop"
    raise ValueError(f"Unknown model label: {model_label}")


def _job_id(stage: str, model_label: str, fold_index: int | None = None) -> str:
    if fold_index is None:
        return f"{stage}.{model_label}"
    return f"{stage}.fold_{fold_index}.{model_label}"


def _fold_row_from_status(job: dict[str, object], fold_index: int, model_label: str) -> dict[str, object]:
    metrics = dict(job["metrics"])
    return {
        "fold": fold_index,
        "family": _model_family(model_label),
        "model": "chemprop" if model_label.startswith("chemprop") else model_label,
        "model_label": model_label,
        **metrics,
    }


def _test_row_from_status(job: dict[str, object], model_label: str) -> dict[str, object]:
    metrics = dict(job["metrics"])
    return {
        "family": _model_family(model_label),
        "model": "chemprop" if model_label.startswith("chemprop") else model_label,
        "model_label": model_label,
        **metrics,
    }


def _evaluate_fold(
    model_label: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict, list[str]]:
    if model_label in classic_model_names():
        train_data = build_feature_dataset(train_df)
        valid_data = build_feature_dataset(valid_df)
        test_data = build_feature_dataset(test_df)
        model, valid_prob = train_classic_and_predict(model_label, train_data, valid_data)
        test_prob = model.predict_proba(test_data.features)[:, 1]
        valid_metrics = compute_metrics(valid_data.labels, valid_prob)
        cv_test_metrics = compute_metrics(test_data.labels, test_prob)
        metrics = {
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **{f"cv_test_{key}": value for key, value in cv_test_metrics.items()},
        }
        invalid_smiles = train_data.invalid_smiles + valid_data.invalid_smiles + test_data.invalid_smiles
    else:
        runner, checkpoints = fit_chemprop_variant(
            model_label=model_label,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_dir,
        )
        valid_prob, _ = runner.predict(
            checkpoint_paths=checkpoints,
            predict_df=valid_df,
            preds_path=output_dir / model_label / "valid_predictions.csv",
        )
        cv_test_prob, _ = runner.predict(
            checkpoint_paths=checkpoints,
            predict_df=test_df,
            preds_path=output_dir / model_label / "cv_test_predictions.csv",
        )
        valid_metrics = compute_metrics(valid_df[config.LABEL_COLUMN].to_numpy(dtype=int), valid_prob)
        cv_test_metrics = compute_metrics(test_df[config.LABEL_COLUMN].to_numpy(dtype=int), cv_test_prob)
        metrics = {
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **{f"cv_test_{key}": value for key, value in cv_test_metrics.items()},
        }
        invalid_smiles = []

    return metrics, invalid_smiles


def _build_cv_metrics(
    benchmark_models: list[str],
    progress: Callable[[str], None] | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    status = load_status()
    fold_rows: list[dict] = []
    invalid_counts: dict[str, int] = {}
    failures: list[str] = []
    cv_test_df = load_test_split()

    for split in load_fold_splits():
        if progress is not None:
            progress(f"[cv] fold_{split.fold_index}")
        for model_label in benchmark_models:
            job_id = _job_id(stage="cv", model_label=model_label, fold_index=split.fold_index)
            existing = get_job(status, job_id)
            if is_completed(existing):
                if progress is not None:
                    progress(f"  - {model_label} [skip completed]")
                fold_rows.append(_fold_row_from_status(existing, split.fold_index, model_label))
                invalid_counts[f"fold_{split.fold_index}_{model_label}"] = int(existing.get("invalid_smiles", 0))
                continue

            if progress is not None:
                progress(f"  - {model_label}")
            try:
                mark_started(status, job_id=job_id, stage="cv", model_label=model_label)
                metrics, invalid_smiles = _evaluate_fold(
                    model_label=model_label,
                    train_df=split.train_df,
                    valid_df=split.valid_df,
                    test_df=cv_test_df,
                    output_dir=config.CHEMPROP_WORK_DIR / "cv" / f"fold_{split.fold_index}",
                )
                mark_completed(
                    status,
                    job_id=job_id,
                    stage="cv",
                    model_label=model_label,
                    metrics=metrics,
                    invalid_smiles=len(invalid_smiles),
                )
                fold_rows.append(
                    {
                        "fold": split.fold_index,
                        "family": _model_family(model_label),
                        "model": "chemprop" if model_label.startswith("chemprop") else model_label,
                        "model_label": model_label,
                        **metrics,
                    }
                )
                invalid_counts[f"fold_{split.fold_index}_{model_label}"] = len(invalid_smiles)
            except Exception as exc:
                mark_failed(status, job_id=job_id, stage="cv", model_label=model_label, error=str(exc))
                failures.append(job_id)
                if progress is not None:
                    progress(f"    failed: {exc}")

    if failures:
        raise RuntimeError(f"CV jobs failed: {', '.join(failures)}")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(config.OUTPUT_DIR / "fold_metrics.csv", index=False)

    summary = (
        fold_metrics.groupby(["family", "model", "model_label"], as_index=False)
        .agg(
            valid_roc_mean=("valid_roc", "mean"),
            valid_roc_std=("valid_roc", "std"),
            valid_prc_mean=("valid_prc", "mean"),
            valid_prc_std=("valid_prc", "std"),
            valid_accuracy_mean=("valid_accuracy", "mean"),
            valid_accuracy_std=("valid_accuracy", "std"),
            valid_f1_mean=("valid_f1", "mean"),
            valid_f1_std=("valid_f1", "std"),
            cv_test_roc_mean=("cv_test_roc", "mean"),
            cv_test_roc_std=("cv_test_roc", "std"),
            cv_test_prc_mean=("cv_test_prc", "mean"),
            cv_test_prc_std=("cv_test_prc", "std"),
            cv_test_accuracy_mean=("cv_test_accuracy", "mean"),
            cv_test_accuracy_std=("cv_test_accuracy", "std"),
            cv_test_f1_mean=("cv_test_f1", "mean"),
            cv_test_f1_std=("cv_test_f1", "std"),
            n_folds=("fold", "count"),
        )
        .sort_values(["cv_test_roc_mean", "valid_roc_mean", "cv_test_prc_mean", "valid_prc_mean"], ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(config.OUTPUT_DIR / "cv5_summary_metrics.csv", index=False)
    if progress is not None:
        progress(f"[cv] wrote {config.OUTPUT_DIR / 'cv5_summary_metrics.csv'}")
    return summary, invalid_counts


def _build_test_metrics(
    benchmark_models: list[str],
    progress: Callable[[str], None] | None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    status = load_status()
    dev_df = load_development_set()
    test_df = load_test_split()
    test_rows: list[dict] = []
    model_paths: dict[str, Path] = {}
    failures: list[str] = []

    dev_data = build_feature_dataset(dev_df)
    test_data = build_feature_dataset(test_df)

    if progress is not None:
        progress("[test] retrain on train+valid and evaluate on hold-out test")

    for model_label in benchmark_models:
        job_id = _job_id(stage="test", model_label=model_label)
        existing = get_job(status, job_id)
        if is_completed(existing):
            if progress is not None:
                progress(f"  - {model_label} [skip completed]")
            existing_model_path = config.ARTIFACTS_DIR / f"{model_label}.joblib"
            if existing_model_path.exists():
                model_paths[model_label] = existing_model_path
            test_rows.append(_test_row_from_status(existing, model_label))
            continue

        if progress is not None:
            progress(f"  - {model_label}")
        try:
            mark_started(status, job_id=job_id, stage="test", model_label=model_label)
            if model_label in classic_model_names():
                model, y_prob = train_classic_and_predict(model_label, dev_data, test_data)
                model_paths[model_label] = save_classic_model(model, model_label, config.ARTIFACTS_DIR)
            else:
                runner, checkpoints = fit_chemprop_variant(
                    model_label=model_label,
                    train_df=dev_df,
                    valid_df=None,
                    test_df=test_df,
                    output_dir=config.CHEMPROP_WORK_DIR / "test",
                )
                y_prob, _ = runner.predict(
                    checkpoint_paths=checkpoints,
                    predict_df=test_df,
                    preds_path=config.CHEMPROP_WORK_DIR / "test" / model_label / "predictions.csv",
                )
                model_paths[model_label] = checkpoints[0]

            metrics = compute_metrics(test_data.labels, y_prob)
            pred_path = write_test_predictions(model_label, test_data.rows, y_prob)
            mark_completed(
                status,
                job_id=job_id,
                stage="test",
                model_label=model_label,
                metrics=metrics,
                invalid_smiles=len(test_data.invalid_smiles),
                extra={"prediction_csv": str(pred_path)},
            )
            test_rows.append(
                {
                    "family": _model_family(model_label),
                    "model": "chemprop" if model_label.startswith("chemprop") else model_label,
                    "model_label": model_label,
                    **metrics,
                }
            )
        except Exception as exc:
            mark_failed(status, job_id=job_id, stage="test", model_label=model_label, error=str(exc))
            failures.append(job_id)
            if progress is not None:
                progress(f"    failed: {exc}")

    if failures:
        raise RuntimeError(f"Test jobs failed: {', '.join(failures)}")

    test_metrics = pd.DataFrame(test_rows).sort_values(["roc", "prc", "f1"], ascending=False)
    test_metrics.to_csv(config.OUTPUT_DIR / "test_metrics.csv", index=False)
    if progress is not None:
        progress(f"[test] wrote {config.OUTPUT_DIR / 'test_metrics.csv'}")
    return test_metrics, model_paths


def run_all_baselines(
    benchmark_models: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    benchmark_models = list(benchmark_models or config.BENCHMARK_MODELS)
    ensure_output_dirs()
    cv_summary, invalid_counts = _build_cv_metrics(
        benchmark_models=benchmark_models,
        progress=progress,
    )
    test_metrics, model_paths = _build_test_metrics(
        benchmark_models=benchmark_models,
        progress=progress,
    )
    cv_summary = cv_summary.merge(
        test_metrics.rename(
            columns={
                "roc": "test_roc",
                "prc": "test_prc",
                "accuracy": "test_accuracy",
                "f1": "test_f1",
            }
        )[["model_label", "test_roc", "test_prc", "test_accuracy", "test_f1"]],
        on="model_label",
        how="left",
    )
    cv_summary.to_csv(config.OUTPUT_DIR / "cv5_summary_metrics.csv", index=False)
    top3_models, best_model = write_rank_outputs(cv_summary=cv_summary, output_dir=config.OUTPUT_DIR)
    write_runtime_config(
        artifacts_dir=config.ARTIFACTS_DIR,
        model_paths=model_paths,
        top_models=top3_models,
        best_model=best_model,
    )

    write_run_config(
        {
            "seed": config.SEED,
            "smiles_col": config.SMILES_COLUMN,
            "label_col": config.LABEL_COLUMN,
            "benchmark_models": benchmark_models,
            "feature": {"type": "morgan", "radius": config.MORGAN_RADIUS, "n_bits": config.MORGAN_BITS},
            "fold_dir": str(config.FOLD_DIR),
            "test_csv": str(config.INPUT_TEST_CSV),
            "ranking_rule": "cv_test > final_test > cv_valid",
            "invalid_smiles": invalid_counts,
        }
    )
    if progress is not None:
        progress(f"[rank] top3: {', '.join(top3_models)}")
        progress(f"[rank] top1: {best_model}")
        progress(f"[artifacts] wrote {config.ARTIFACTS_DIR / 'runtime_config.json'}")

    return {
        "output_dir": config.OUTPUT_DIR,
        "cv_summary_csv": config.OUTPUT_DIR / "cv5_summary_metrics.csv",
        "test_metrics_csv": config.OUTPUT_DIR / "test_metrics.csv",
        "runtime_config_json": config.ARTIFACTS_DIR / "runtime_config.json",
        "top3_models": top3_models,
        "best_model": best_model,
    }
