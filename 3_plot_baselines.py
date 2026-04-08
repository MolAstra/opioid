#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/baselines")
FIGURES_DIR = OUTPUT_DIR / "figures"
SUMMARY_CSV = OUTPUT_DIR / "cv5_summary_metrics.csv"

MODELS = {
    "rf": "RF",
    "svm": "SVM",
    "xgb": "XGB",
    "lgbm": "LightGBM",
    "tabpfn": "TabPFN",
    "chemprop_morgan_rdkit2d": "Chemprop",
}

METRICS = {
    "roc": ("AUROC", "baseline_auroc_bars.png"),
    "prc": ("AUPRC", "baseline_auprc_bars.png"),
    "accuracy": ("Accuracy", "baseline_accuracy_bars.png"),
    "f1": ("F1", "baseline_f1_bars.png"),
}

SPLITS = [
    ("valid", "CV valid"),
    ("cv_test", "CV test"),
    ("test", "Test"),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["model_label"].isin(MODELS)].copy()
    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=MODELS.keys(),
        ordered=True,
    )
    df["display_name"] = df["model_label"].map(MODELS)
    return df.sort_values("model_label").reset_index(drop=True)


def get_metric_column(split: str, metric: str) -> str:
    return f"test_{metric}" if split == "test" else f"{split}_{metric}_mean"


def get_error_column(split: str, metric: str) -> str | None:
    if split == "test":
        return None
    return f"{split}_{metric}_std"


def add_value_labels(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    errors: np.ndarray | None,
    series_idx: int,
    fmt: str = "{:.2f}",
) -> None:
    offsets = [0.012, 0.012, 0.012]
    y_offset = offsets[series_idx]

    if errors is None:
        errors = np.zeros_like(values)

    for bar, value, error in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    x = np.arange(len(df))
    width = 0.22
    colors = plt.get_cmap("Set1").colors

    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    max_val = 0.0
    for i, (split, label) in enumerate(SPLITS):
        values = df[get_metric_column(split, metric)].to_numpy(dtype=float)
        error_column = get_error_column(split, metric)
        errors = None if error_column is None else df[error_column].to_numpy(dtype=float)
        upper_values = values if errors is None else values + errors
        max_val = max(max_val, upper_values.max())

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=label,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
            yerr=errors,
            ecolor="#2F2F2F",
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        add_value_labels(ax, bars, values, errors, series_idx=i)

    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"])
    ax.set_ylabel(ylabel)

    upper = min(1.18, max(1.0, max_val + 0.10))
    ax.set_ylim(0, upper)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linewidth=0.8, alpha=0.3)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.80)

    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_summary()
    for metric, (ylabel, filename) in METRICS.items():
        plot_metric(df, metric, ylabel, filename)


if __name__ == "__main__":
    main()
