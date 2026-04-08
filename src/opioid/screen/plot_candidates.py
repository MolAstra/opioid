from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs/candidates")
FIGURES_DIR = OUTPUT_DIR / "figures"
STATS_CSV = OUTPUT_DIR / "candidate_stats.csv"
SUMMARY_CSV = OUTPUT_DIR / "candidate_plot_summary.csv"

DB_ORDER = ["gpcrdb", "zinc", "reinvent", "ouroboros"]
DB_LABELS = {
    "gpcrdb": "GPCRdb",
    "zinc": "ZINC",
    "reinvent": "REINVENT",
    "ouroboros": "OUROBOROS",
}
DB_COLORS = {
    "gpcrdb": "#b44b2a",
    "zinc": "#5f7f95",
    "reinvent": "#7a9f61",
    "ouroboros": "#c28b2c",
}

HIT_RATE_PNG = FIGURES_DIR / "candidate_hit_rate_bars.png"
COUNT_PNG = FIGURES_DIR / "candidate_count_bars.png"
ECDF_PNG = FIGURES_DIR / "candidate_top3_prob_ecdf.png"
TAIL_PNG = FIGURES_DIR / "candidate_top_tail_box_or_violin.png"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_stats() -> pd.DataFrame:
    df = pd.read_csv(STATS_CSV)
    df["db"] = pd.Categorical(df["db"], categories=DB_ORDER, ordered=True)
    df = df.sort_values("db").reset_index(drop=True)
    df["display_name"] = df["db"].map(DB_LABELS)
    return df


def load_predictions() -> dict[str, pd.DataFrame]:
    data = {}
    for db in DB_ORDER:
        df = pd.read_csv(OUTPUT_DIR / f"pred_{db}.csv")
        df["db"] = db
        data[db] = df
    return data


def build_summary_table(stats_df: pd.DataFrame, pred_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in stats_df.iterrows():
        db = str(row["db"])
        df = pred_map[db]
        q = df["top3_mean_prob"].quantile([0.5, 0.9, 0.99]).to_dict()
        rows.append(
            {
                "db": db,
                "n_valid": int(row["n_valid"]),
                "top3_all1_n": int(row["top3_all1_n"]),
                "top3_all1_rate": float(row["top3_all1_rate"]),
                "top3_mean_prob_median": float(q[0.5]),
                "top3_mean_prob_q90": float(q[0.9]),
                "top3_mean_prob_q99": float(q[0.99]),
                "top3_mean_prob_max": float(df["top3_mean_prob"].max()),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    return summary_df


def plot_hit_rate(stats_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(stats_df))
    colors = [DB_COLORS[str(db)] for db in stats_df["db"]]
    bars = ax.bar(x, stats_df["top3_all1_rate"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["display_name"])
    ax.set_ylabel("Top-3 consensus rate")
    ax.set_ylim(0, max(stats_df["top3_all1_rate"].max() * 1.18, 0.08))
    ax.grid(axis="y", linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, stats_df["top3_all1_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(HIT_RATE_PNG, bbox_inches="tight")
    plt.close(fig)


def plot_counts(stats_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(stats_df))
    width = 0.34
    ax.bar(x - width / 2, stats_df["n_valid"], width=width, color="#cfd7df", edgecolor="white", linewidth=0.8, label="Valid")
    ax.bar(x + width / 2, stats_df["top3_all1_n"], width=width, color="#b44b2a", edgecolor="white", linewidth=0.8, label="Top-3 consensus")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["display_name"])
    ax.set_ylabel("Molecule count")
    ax.grid(axis="y", linewidth=0.8, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(COUNT_PNG, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(pred_map: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for db in DB_ORDER:
        df = pred_map[db]
        values = df["top3_mean_prob"].sort_values().reset_index(drop=True)
        y = (values.index + 1) / len(values)
        ax.plot(values.to_numpy(dtype=float), y.to_numpy(dtype=float), linewidth=2.0, color=DB_COLORS[db], label=DB_LABELS[db])
    ax.set_xlabel("Top-3 mean probability")
    ax.set_ylabel("ECDF")
    ax.grid(linewidth=0.8, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ECDF_PNG, bbox_inches="tight")
    plt.close(fig)


def plot_top_tail(pred_map: dict[str, pd.DataFrame]) -> None:
    tail_values = []
    tail_labels = []
    tail_colors = []
    for db in DB_ORDER:
        df = pred_map[db].sort_values("top3_mean_prob", ascending=False)
        tail_n = max(int(np.ceil(len(df) * 0.01)), 10)
        tail = df.head(tail_n)["top3_mean_prob"].to_numpy(dtype=float)
        tail_values.append(tail)
        tail_labels.append(DB_LABELS[db])
        tail_colors.append(DB_COLORS[db])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    violin = ax.violinplot(tail_values, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], tail_colors):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.85)
    violin["cmedians"].set_color("#1f1f1f")
    violin["cmedians"].set_linewidth(1.0)
    ax.set_xticks(np.arange(1, len(tail_labels) + 1))
    ax.set_xticklabels(tail_labels)
    ax.set_ylabel("Top-1% top-3 mean probability")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(TAIL_PNG, bbox_inches="tight")
    plt.close(fig)


def run_candidate_plots() -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    stats_df = load_stats()
    pred_map = load_predictions()
    build_summary_table(stats_df, pred_map)
    plot_hit_rate(stats_df)
    plot_counts(stats_df)
    plot_ecdf(pred_map)
    plot_top_tail(pred_map)
    return {
        "summary_csv": SUMMARY_CSV,
        "hit_rate_png": HIT_RATE_PNG,
        "count_png": COUNT_PNG,
        "ecdf_png": ECDF_PNG,
        "tail_png": TAIL_PNG,
    }
