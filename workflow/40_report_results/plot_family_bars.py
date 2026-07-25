#!/usr/bin/env python3
"""Draw a validation-selected, model-family-level descriptive figure set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Final
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opioid.benchmark.family_selection import (
    MODEL_FAMILIES,
    model_family,
    select_validation_representatives,
)

WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = WORKFLOW_DIR / "results" / "all_candidates.csv"
DEFAULT_OUTPUT_DIR = WORKFLOW_DIR / "results" / "model_family_bars"
FIGURE_SIZE: Final = (15.5, 7.2)
FIGURE_DPI: Final = 600
FONT_FAMILY: Final = "Arial"
LEGEND_COLUMNS: Final = 3
VALUE_LABEL_DECIMALS: Final = 2
MODEL_SPACING: Final = 0.93
BAR_WIDTH: Final = 0.135

PROTOCOLS: Final = ("scaffold", "random", "datasail")
PROTOCOL_LABELS: Final = {
    "scaffold": "Scaffold",
    "random": "Random",
    "datasail": "DataSAIL",
}
METRICS: Final = {
    "roc": "AUROC",
    "prc": "AUPRC",
    "accuracy": "Accuracy",
    "f1": "F1",
}
MODEL_TICK_LABELS: Final = {
    "SVM": "SVM",
    "RF": "RF",
    "XGB": "XGB",
    "LightGBM": "LightGBM",
    "TabPFN": "TabPFN",
    "Chemprop": "Chemprop",
    "KNN": "KNN",
    "LR": "LR",
}
BAR_STYLES: Final = {
    ("scaffold", "valid"): ("#62C7F2", "Scaffold validation"),
    ("scaffold", "test"): ("#0077B6", "Scaffold held-out test"),
    ("random", "valid"): ("#FFB347", "Random validation"),
    ("random", "test"): ("#F05A00", "Random held-out test"),
    ("datasail", "valid"): ("#7DDB8A", "DataSAIL validation"),
    ("datasail", "test"): ("#168B54", "DataSAIL held-out test"),
}
IDENTITY_COLUMNS: Final = {
    "protocol",
    "family",
    "algorithm",
    "preset",
    "representation",
    "candidate_id",
    "model_name",
    "model_type",
    "n_folds",
    "test_n_folds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four descriptive model-family bar charts. For every metric, protocol, "
            "and family, the displayed candidate is fixed by the highest validation AUROC."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Workflow 40 all_candidates.csv (default: results/all_candidates.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dedicated derived-output directory (default: results/model_family_bars).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing model-family figure directory.",
    )
    return parser.parse_args()


def load_candidates(path: Path | str) -> pd.DataFrame:
    """Load and validate the complete consolidated Workflow 40 candidate table."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing Workflow 40 candidate table: {source}")
    frame = pd.read_csv(source)
    required = set(IDENTITY_COLUMNS)
    required.update(
        f"{stage}_{metric}_{stat}"
        for stage in ("valid", "test")
        for metric in METRICS
        for stat in ("mean", "std")
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {missing}")
    if frame.duplicated(["protocol", "candidate_id"]).any():
        raise ValueError("Candidate table contains duplicate protocol/candidate rows")
    if set(frame["protocol"].astype(str)) != set(PROTOCOLS):
        raise ValueError(f"Candidate table must contain exactly these protocols: {PROTOCOLS}")
    for fold_column in ("n_folds", "test_n_folds"):
        if not pd.to_numeric(frame[fold_column], errors="coerce").eq(5).all():
            raise ValueError(f"Candidate table requires {fold_column}=5 for every row")
    numeric_columns = [
        f"{stage}_{metric}_{stat}"
        for stage in ("valid", "test")
        for metric in METRICS
        for stat in ("mean", "std")
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Candidate table contains non-finite metric means or SD values")
    if (numeric[[column for column in numeric if column.endswith("_std")]] < 0).any().any():
        raise ValueError("Candidate table contains a negative SD")
    frame = frame.copy()
    frame["model_family"] = frame["algorithm"].astype(str).map(model_family)
    if set(frame["model_family"]) != set(MODEL_FAMILIES):
        raise ValueError("Candidate table does not contain the expected eight model families")
    candidate_sets = {
        protocol: frozenset(
            frame.loc[frame["protocol"] == protocol, "candidate_id"].astype(str)
        )
        for protocol in PROTOCOLS
    }
    if len(set(candidate_sets.values())) != 1:
        raise ValueError("All protocols must contain the same registered candidate catalog")
    return frame


def select_family_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one validation-AUROC candidate per protocol and family for every metric."""

    rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        representatives = select_validation_representatives(frame, protocol=protocol)
        for metric, metric_label in METRICS.items():
            for family in MODEL_FAMILIES:
                selected = representatives.loc[
                    representatives["model_family"] == family
                ].iloc[0]
                rows.append(
                    {
                        "metric": metric_label,
                        "metric_key": metric,
                        "protocol": protocol,
                        "model_family": family,
                        "selected_candidate_id": selected["candidate_id"],
                        "selected_model_name": selected["model_name"],
                        "selected_model_type": selected["model_type"],
                        "selected_algorithm": selected["algorithm"],
                        "selected_preset": selected["preset"],
                        "selected_representation": selected["representation"],
                        "selection_stage": "validation",
                        "selection_statistic": "mean",
                        "selection_column": "valid_roc_mean",
                        "selection_value": float(selected["valid_roc_mean"]),
                        "valid_mean": float(selected[f"valid_{metric}_mean"]),
                        "valid_std": float(selected[f"valid_{metric}_std"]),
                        "test_mean": float(selected[f"test_{metric}_mean"]),
                        "test_std": float(selected[f"test_{metric}_std"]),
                    }
                )
    selected = pd.DataFrame(rows)
    expected_rows = len(METRICS) * len(PROTOCOLS) * len(MODEL_FAMILIES)
    if len(selected) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} selections, observed {len(selected)}")
    return selected


def plot_metric(
    selected: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
) -> Path:
    """Draw one 8-family × 6-bar validation/shared-test comparison."""

    metric_label = METRICS[metric]
    metric_rows = selected.loc[selected["metric_key"] == metric]
    x = np.arange(len(MODEL_FAMILIES), dtype=float) * MODEL_SPACING
    combinations = tuple(BAR_STYLES)
    offsets = (
        np.arange(len(combinations), dtype=float) - (len(combinations) - 1) / 2
    ) * BAR_WIDTH

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 12,
            "font.weight": "normal",
            "axes.labelsize": 15,
            "axes.labelweight": "normal",
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for offset, (protocol, stage) in zip(offsets, combinations, strict=True):
        ordered = (
            metric_rows.loc[metric_rows["protocol"] == protocol]
            .set_index("model_family")
            .loc[list(MODEL_FAMILIES)]
        )
        means = ordered[f"{stage}_mean"].to_numpy(dtype=float)
        errors = ordered[f"{stage}_std"].to_numpy(dtype=float)
        color, legend_label = BAR_STYLES[(protocol, stage)]
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            yerr=errors,
            capsize=2.2,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.6,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#37474F"},
            label=legend_label,
            zorder=3,
        )
        for position, mean, error in zip(x + offset, means, errors, strict=True):
            ax.annotate(
                f"{mean:.{VALUE_LABEL_DECIMALS}f}",
                xy=(position, mean + error),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="normal",
                color="#202124",
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_TICK_LABELS[family] for family in MODEL_FAMILIES])
    ax.set_ylabel(metric_label)
    ax.set_ylim(0.0, 1.12)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.8, alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#90A4AE")
    ax.spines["bottom"].set_color("#90A4AE")
    ax.tick_params(axis="both", colors="#263238")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=LEGEND_COLUMNS,
        frameon=False,
    )
    fig.tight_layout(rect=(0.01, 0.01, 0.995, 0.87))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_package(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    replace: bool = False,
) -> dict[str, Path]:
    """Atomically create the derived model-family figure package."""

    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and not replace:
        raise FileExistsError(
            f"Model-family output already exists: {destination}. Re-run with --replace."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = load_candidates(source)
    selected = select_family_candidates(frame)
    staging = Path(tempfile.mkdtemp(prefix=".model-family-bars-", dir=destination.parent))
    backup = destination.parent / f".model-family-bars-backup-{uuid.uuid4().hex}"
    try:
        outputs: dict[str, Path] = {}
        selection_path = staging / "selected_candidates.csv"
        selected.to_csv(selection_path, index=False)
        outputs["selected_candidates"] = selection_path
        for metric, label in METRICS.items():
            figure_path = staging / f"{label.casefold()}.png"
            outputs[label.casefold()] = plot_metric(
                selected,
                metric=metric,
                output_path=figure_path,
            )
        created_at = datetime.now(timezone.utc).isoformat()
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest(source, staging, frame, selected, created_at),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs["manifest"] = manifest_path
        if destination.exists():
            destination.rename(backup)
        staging.rename(destination)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if not destination.exists() and backup.exists():
            backup.rename(destination)
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {name: destination / path.name for name, path in outputs.items()}


def _manifest(
    source: Path,
    staging: Path,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    created_at: str,
) -> dict[str, object]:
    candidate_counts = (
        frame.loc[frame["protocol"] == PROTOCOLS[0]]
        .groupby("model_family", sort=False)["candidate_id"]
        .nunique()
        .reindex(MODEL_FAMILIES)
    )
    files = {
        path.name: _sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "kind": "muorscreen.validation_selected_family_bar_figures",
        "schema_version": 2,
        "created_at": created_at,
        "selection_scope": "validation_selected_descriptive_family_view",
        "independent_evaluation": False,
        "deployment_artifact": False,
        "warning": (
            "One candidate per protocol and model family is selected by validation AUROC. Every "
            "metric displays validation and held-out-test values from that same frozen candidate. "
            "The figures are descriptive and are not a deployment selection."
        ),
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "rows": int(len(frame)),
            "candidates_per_protocol": int(frame["candidate_id"].nunique()),
        },
        "protocols": list(PROTOCOLS),
        "metrics": list(METRICS.values()),
        "model_families": list(MODEL_FAMILIES),
        "family_candidate_counts": {
            family: int(candidate_counts.loc[family]) for family in MODEL_FAMILIES
        },
        "selection_rule": {
            "group_by": ["protocol", "model_family"],
            "maximize": "valid_roc_mean",
            "tie_breaker": "candidate_id ascending",
            "paired_test": "held-out-test mean/SD from the same selected candidate",
            "preset_and_representation_selected_together": True,
        },
        "selection_rows": int(len(selected)),
        "bars_per_figure": len(PROTOCOLS) * 2 * len(MODEL_FAMILIES),
        "presentation": {
            "figure_size_inches": list(FIGURE_SIZE),
            "dpi": FIGURE_DPI,
            "font_family": FONT_FAMILY,
            "font_weight": "normal",
            "legend_columns": LEGEND_COLUMNS,
            "value_label_decimals": VALUE_LABEL_DECIMALS,
            "model_spacing": MODEL_SPACING,
            "bar_width": BAR_WIDTH,
            "palette": {
                f"{protocol}_{stage}": color
                for (protocol, stage), (color, _) in BAR_STYLES.items()
            },
        },
        "files": files,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    outputs = write_package(args.input, args.output_dir, replace=args.replace)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
