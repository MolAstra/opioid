"""Unified publication figures for the complete reviewer candidate matrix."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from opioid.benchmark.context import DEFAULT_OUTPUT_ROOT, EVALUATION_PROTOCOLS
from opioid.benchmark.display_names import add_model_display_columns


CV_METRICS: dict[str, str] = {
    "roc": "AUROC",
    "prc": "AUPRC",
    "accuracy": "Accuracy",
    "f1": "F1",
    "mcc": "MCC",
}
STAGE_LABELS = {"valid": "Five-fold validation", "test": "Shared test"}
PROTOCOL_LABELS = {"scaffold": "Scaffold", "random": "Random", "datasail": "DataSAIL"}
REQUIRED_COLUMNS = frozenset(
    {
        "family",
        "algorithm",
        "preset",
        "representation",
        "candidate_id",
        "status",
        *(f"valid_{metric}_mean" for metric in CV_METRICS),
    }
)
COMPLETED_STATUSES = frozenset({"complete", "completed"})
UNIFIED_COLOR = "#356D9A"


def set_style() -> None:
    """Apply the shared 300-dpi, white-background publication style."""

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def benchmark_dir(protocol: str, output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> Path:
    """Return the benchmark artifact root for one protocol."""

    if protocol not in EVALUATION_PROTOCOLS:
        expected = ", ".join(EVALUATION_PROTOCOLS)
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {expected}")
    return Path(output_root) / protocol / "benchmark"


def load_summary(summary_csv: Path | str) -> pd.DataFrame:
    """Load completed candidates with finite five-fold validation metrics."""

    path = Path(summary_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark summary: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    work = frame.copy()
    for column in ("family", "algorithm", "preset", "representation", "candidate_id", "status"):
        work[column] = work[column].fillna("").astype(str).str.strip()
    for stage in ("valid", "test"):
        for metric in CV_METRICS:
            for stat in ("mean", "std"):
                column = f"{stage}_{metric}_{stat}"
                if column in work:
                    work[column] = pd.to_numeric(work[column], errors="coerce")
    valid_columns = [f"valid_{metric}_mean" for metric in CV_METRICS]
    complete = work["status"].str.casefold().isin(COMPLETED_STATUSES)
    finite = np.isfinite(work[valid_columns].to_numpy(dtype=float)).all(axis=1)
    return add_model_display_columns(work.loc[complete & finite].reset_index(drop=True))


def load_protocol_frames(
    input_root: Path | str = DEFAULT_OUTPUT_ROOT,
    protocols: Iterable[str] = EVALUATION_PROTOCOLS,
) -> dict[str, pd.DataFrame]:
    """Load summaries and require an identical candidate catalog across protocols."""

    root = Path(input_root)
    selected = tuple(protocols)
    if not selected:
        raise ValueError("At least one protocol is required")
    frames = {
        protocol: load_summary(benchmark_dir(protocol, root) / "cv_summary_metrics.csv")
        for protocol in selected
    }
    expected_ids = set(next(iter(frames.values()))["candidate_id"])
    if any(set(frame["candidate_id"]) != expected_ids for frame in frames.values()):
        raise ValueError("Figure generation requires the same complete candidate set")
    return frames


def validation_candidate_order(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Order candidates only by mean validation AUROC across available protocols."""

    ordering = (
        pd.concat(
            [frame[["candidate_id", "valid_roc_mean"]] for frame in frames.values()],
            ignore_index=True,
        )
        .groupby("candidate_id", as_index=False)["valid_roc_mean"]
        .mean()
    )
    return ordering.sort_values(
        ["valid_roc_mean", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )["candidate_id"].tolist()


def write_selected_figures(
    frames: Mapping[str, pd.DataFrame],
    figures_dir: Path | str,
    *,
    stages: Iterable[str] = ("valid", "test"),
) -> dict[str, Path]:
    """Write the compact formal figure set for the supplied protocols and stages."""

    set_style()
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_stages = _normalize_stages(stages)
    order = validation_candidate_order(frames)
    outputs: dict[str, Path] = {}

    if tuple(frames) == tuple(EVALUATION_PROTOCOLS):
        for stage in selected_stages:
            for metric in CV_METRICS:
                key = f"{metric}_{stage}"
                outputs[key] = plot_metric_by_split(
                    frames,
                    order,
                    stage=stage,
                    metric=metric,
                    output_path=destination / f"{key}.png",
                )
        if set(selected_stages) == {"valid", "test"}:
            for metric in CV_METRICS:
                key = f"{metric}_validation_vs_test"
                outputs[key] = plot_validation_vs_test(
                    frames,
                    metric=metric,
                    output_path=destination / f"{key}.png",
                )

    for protocol, frame in frames.items():
        for stage in selected_stages:
            key = f"{protocol}_{stage}_overview"
            outputs[key] = plot_protocol_overview(
                frame,
                order,
                protocol=protocol,
                stage=stage,
                output_path=destination / f"{key}.png",
            )
    return outputs


def plot_metric_by_split(
    frames: Mapping[str, pd.DataFrame],
    candidate_order: list[str],
    *,
    stage: str,
    metric: str,
    output_path: Path,
) -> Path:
    """Plot one annotated candidate × split-method metric heatmap."""

    _validate_stage_metric(frames, stage, metric)
    protocols = tuple(frames)
    matrix = np.column_stack(
        [
            frames[protocol]
            .set_index("candidate_id")
            .loc[candidate_order, f"{stage}_{metric}_mean"]
            .to_numpy(dtype=float)
            for protocol in protocols
        ]
    )
    exemplar = next(iter(frames.values())).set_index("candidate_id")
    labels = [_candidate_label(exemplar.loc[candidate_id]) for candidate_id in candidate_order]
    norm, cmap = _metric_color(metric)
    fig, ax = plt.subplots(
        figsize=(8.8, max(8.0, 0.35 * len(candidate_order) + 2.0)),
    )
    image = ax.imshow(matrix, norm=norm, cmap=cmap, aspect="auto")
    ax.set_title(f"{CV_METRICS[metric]} — {STAGE_LABELS[stage]}")
    ax.set_xticks(np.arange(len(protocols)))
    ax.set_xticklabels([PROTOCOL_LABELS[item] for item in protocols])
    ax.set_yticks(np.arange(len(candidate_order)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=7.5, pad=2)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=_annotation_color(cmap, norm, value),
                fontsize=7.2,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label(CV_METRICS[metric])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_validation_vs_test(
    frames: Mapping[str, pd.DataFrame],
    *,
    metric: str,
    output_path: Path,
) -> Path:
    """Plot validation/test agreement without model-family grouping."""

    _validate_stage_metric(frames, "valid", metric)
    _validate_stage_metric(frames, "test", metric)
    protocols = tuple(frames)
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.2 * len(protocols), 4.8), sharex=True)
    axes_array = np.atleast_1d(axes)
    lower, upper = (-1.0, 1.0) if metric == "mcc" else (0.0, 1.0)
    for ax, protocol in zip(axes_array, protocols, strict=True):
        frame = frames[protocol]
        valid = frame[f"valid_{metric}_mean"].to_numpy(dtype=float)
        test = frame[f"test_{metric}_mean"].to_numpy(dtype=float)
        correlation = float(np.corrcoef(valid, test)[0, 1])
        ax.scatter(valid, test, s=24, color=UNIFIED_COLOR, alpha=0.78, edgecolors="none")
        ax.plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.set_xlabel(f"Validation {CV_METRICS[metric]}")
        ax.text(
            0.04,
            0.96,
            f"Pearson r = {correlation:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.grid(linewidth=0.5, alpha=0.25)
    axes_array[0].set_ylabel(f"Shared-test {CV_METRICS[metric]}")
    fig.suptitle(f"Validation vs shared-test {CV_METRICS[metric]}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_protocol_overview(
    frame: pd.DataFrame,
    candidate_order: list[str],
    *,
    protocol: str,
    stage: str,
    output_path: Path,
) -> Path:
    """Plot a five-metric mean ± SD overview for one protocol and stage."""

    for metric in CV_METRICS:
        _require_metric_columns(frame, stage, metric)
    ordered = frame.set_index("candidate_id").loc[candidate_order]
    labels = [_candidate_label(row) for _, row in ordered.iterrows()]
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(
        1,
        len(CV_METRICS),
        figsize=(24.0, max(8.0, 0.35 * len(ordered) + 2.0)),
        sharey=True,
    )
    for index, (ax, (metric, label)) in enumerate(zip(axes, CV_METRICS.items(), strict=True)):
        ax.errorbar(
            ordered[f"{stage}_{metric}_mean"].to_numpy(dtype=float),
            y,
            xerr=ordered[f"{stage}_{metric}_std"].to_numpy(dtype=float),
            fmt="o",
            color=UNIFIED_COLOR,
            markersize=3.5,
            capsize=2.0,
            elinewidth=0.8,
        )
        ax.set_title(label)
        ax.set_xlim(-1.03, 1.03) if metric == "mcc" else ax.set_xlim(0.0, 1.03)
        ax.grid(axis="x", linewidth=0.6, alpha=0.28)
        if index == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.tick_params(axis="y", labelsize=7.5, pad=2)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle(f"{PROTOCOL_LABELS[protocol]} — {STAGE_LABELS[stage]}", y=0.995)
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.988))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _candidate_label(row: pd.Series) -> str:
    return f"{row['model_name']}\n{row['model_type']}"


def _normalize_stages(stages: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(stage) for stage in stages))
    if not selected:
        raise ValueError("At least one stage is required")
    unknown = set(selected).difference(STAGE_LABELS)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return selected


def _validate_stage_metric(frames: Mapping[str, pd.DataFrame], stage: str, metric: str) -> None:
    if stage not in STAGE_LABELS:
        raise ValueError(f"Unknown stage {stage!r}")
    if metric not in CV_METRICS:
        raise ValueError(f"Unknown metric {metric!r}")
    for frame in frames.values():
        _require_metric_columns(frame, stage, metric)


def _require_metric_columns(frame: pd.DataFrame, stage: str, metric: str) -> None:
    columns = [f"{stage}_{metric}_mean", f"{stage}_{metric}_std"]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot plot {stage} {metric}; missing columns: {missing}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Cannot plot {stage} {metric} with non-finite values")


def _metric_color(metric: str) -> tuple[Normalize, object]:
    if metric == "mcc":
        return Normalize(vmin=-1.0, vmax=1.0), plt.colormaps["RdBu_r"]
    return Normalize(vmin=0.0, vmax=1.0), plt.colormaps["cividis"]


def _annotation_color(cmap: object, norm: Normalize, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))  # type: ignore[operator]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance > 0.55 else "#FFFFFF"


__all__ = [
    "CV_METRICS",
    "benchmark_dir",
    "load_protocol_frames",
    "load_summary",
    "plot_metric_by_split",
    "plot_protocol_overview",
    "plot_validation_vs_test",
    "validation_candidate_order",
    "write_selected_figures",
]
