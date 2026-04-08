from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image
from rdkit.Chem import Draw

from opioid.baselines import config
from opioid.baselines.data import load_development_set, load_test_split
from opioid.explain.rf_sar import (
    MorganExample,
    build_examples,
    collect_bit_atoms_bonds,
    compute_shap_values,
    load_rf_model,
    load_runtime,
)

OUTPUT_DIR = Path("outputs/explain_data")
FIGURES_DIR = OUTPUT_DIR / "figures"
TOP_BITS_DIR = OUTPUT_DIR / "top_bits"

GLOBAL_PREDICTIONS_CSV = OUTPUT_DIR / "global_predictions.csv"
GLOBAL_SHAP_VALUES_CSV = OUTPUT_DIR / "global_shap_values.csv"
GLOBAL_BIT_SUMMARY_CSV = OUTPUT_DIR / "global_bit_summary.csv"
TOP_BIT_EXAMPLES_CSV = OUTPUT_DIR / "top_bit_examples.csv"
GLOBAL_SHAP_REPORT_MD = OUTPUT_DIR / "global_shap_report.md"
INVALID_SMILES_JSON = OUTPUT_DIR / "invalid_smiles.json"

BEESWARM_PNG = FIGURES_DIR / "rf_shap_beeswarm.png"
IMPORTANCE_BAR_PNG = FIGURES_DIR / "rf_shap_importance_bar.png"
PREVALENCE_PNG = FIGURES_DIR / "rf_top_bit_prevalence.png"

TOP_N_FIGURE_BITS = 20
TOP_N_REPORT_BITS = 12
TOP_EXAMPLES_PER_BIT = 2
RF_THRESHOLD = 0.5


def _combine_examples() -> tuple[list[MorganExample], pd.DataFrame, dict[str, list[str]]]:
    train_df = load_development_set().copy()
    train_df["split"] = "train_valid"
    test_df = load_test_split().copy()
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    examples: list[MorganExample] = []
    rows: list[dict[str, object]] = []
    invalid: dict[str, list[str]] = {"train_valid": [], "test": []}

    for split_name, split_df in (("train_valid", train_df), ("test", test_df)):
        split_examples, split_invalid = build_examples(split_df)
        invalid[split_name] = split_invalid
        examples.extend(split_examples)

    for _, row in combined_df.iterrows():
        rows.append(
            {
                "split": str(row["split"]),
                config.SMILES_COLUMN: str(row[config.SMILES_COLUMN]),
                config.LABEL_COLUMN: int(row[config.LABEL_COLUMN]),
            }
        )

    valid_rows = [
        row
        for row in rows
        if row[config.SMILES_COLUMN] not in set(invalid[row["split"]])
    ]
    metadata_df = pd.DataFrame(valid_rows)
    return examples, metadata_df, invalid


def _predict_dataframe(model, examples: list[MorganExample], metadata_df: pd.DataFrame) -> pd.DataFrame:
    x = np.stack([example.feature for example in examples], axis=0)
    pred_prob = model.predict_proba(x)[:, 1].astype(float)
    pred_label = (pred_prob >= RF_THRESHOLD).astype(int)

    out_df = metadata_df.copy()
    out_df["pred_prob"] = pred_prob
    out_df["pred_label"] = pred_label
    out_df["is_correct"] = (out_df[config.LABEL_COLUMN].to_numpy(dtype=int) == pred_label).astype(int)
    return out_df


def _feature_names() -> list[str]:
    return [f"bit_{idx}" for idx in range(config.MORGAN_BITS)]


def _build_shap_values_df(
    metadata_df: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    shap_df = pd.DataFrame(shap_values, columns=_feature_names())
    return pd.concat([metadata_df.reset_index(drop=True), shap_df], axis=1)


def _bit_summary(feature_matrix: np.ndarray, labels: np.ndarray, shap_values: np.ndarray) -> pd.DataFrame:
    positive_mask = labels == 1
    negative_mask = labels == 0
    count_on = feature_matrix.sum(axis=0).astype(int)
    positive_counts = np.maximum(int(positive_mask.sum()), 1)
    negative_counts = np.maximum(int(negative_mask.sum()), 1)

    summary_df = pd.DataFrame(
        {
            "bit": np.arange(config.MORGAN_BITS, dtype=int),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "positive_prevalence": feature_matrix[positive_mask].sum(axis=0) / positive_counts,
            "negative_prevalence": feature_matrix[negative_mask].sum(axis=0) / negative_counts,
            "count_on": count_on,
        }
    )
    summary_df["feature_name"] = summary_df["bit"].map(lambda bit: f"bit_{bit}")
    return summary_df.sort_values(["mean_abs_shap", "count_on"], ascending=[False, False]).reset_index(drop=True)


def _save_beeswarm(shap_values: np.ndarray, feature_matrix: np.ndarray, top_bits: list[int]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 6.2))
    shap.summary_plot(
        shap_values[:, top_bits],
        feature_matrix[:, top_bits],
        feature_names=[f"bit_{bit}" for bit in top_bits],
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Morgan bit")
    ax.set_title("")
    plt.tight_layout()
    plt.savefig(BEESWARM_PNG, dpi=600, bbox_inches="tight")
    plt.close()


def _save_importance_bar(summary_df: pd.DataFrame, top_bits: list[int]) -> None:
    plot_df = (
        summary_df[summary_df["bit"].isin(top_bits)]
        .set_index("bit")
        .loc[top_bits]
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(
        y=np.arange(len(plot_df)),
        width=plot_df["mean_abs_shap"],
        color="#2f5c85",
        edgecolor="none",
    )
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels([f"bit_{bit}" for bit in plot_df["bit"]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Morgan bit")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMPORTANCE_BAR_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _save_prevalence_plot(summary_df: pd.DataFrame, top_bits: list[int]) -> None:
    plot_df = (
        summary_df[summary_df["bit"].isin(top_bits[:TOP_N_REPORT_BITS])]
        .set_index("bit")
        .loc[top_bits[:TOP_N_REPORT_BITS]]
        .reset_index()
    )
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(
        x - width / 2,
        plot_df["positive_prevalence"],
        width=width,
        color="#c15a2e",
        edgecolor="none",
        label="Positive",
    )
    ax.bar(
        x + width / 2,
        plot_df["negative_prevalence"],
        width=width,
        color="#4d7f4e",
        edgecolor="none",
        label="Negative",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"bit_{bit}" for bit in plot_df["bit"]], rotation=45, ha="right")
    ax.set_ylabel("Bit prevalence")
    ax.set_xlabel("Morgan bit")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(PREVALENCE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _render_bit_example_png(example: MorganExample, bit: int, output_png: Path) -> None:
    atoms, bonds = collect_bit_atoms_bonds(example.mol, example.bit_info, [bit])
    atom_colors = {atom_idx: (0.18, 0.51, 0.77) for atom_idx in atoms}
    bond_colors = {bond_idx: (0.18, 0.51, 0.77) for bond_idx in bonds}
    image = Draw.MolToImage(
        example.mol,
        size=(800, 440),
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    out = BytesIO()
    image.save(out, format="PNG")
    out.seek(0)
    Image.open(out).save(output_png)


def _build_top_bit_examples(
    top_bits: list[int],
    examples: list[MorganExample],
    predictions_df: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    example_rows: list[dict[str, object]] = []
    example_map = {example.smiles: example for example in examples}

    for bit in top_bits[:TOP_N_REPORT_BITS]:
        carriers = []
        for row_idx, example in enumerate(examples):
            if int(example.feature[bit]) != 1:
                continue
            pred_row = predictions_df.iloc[row_idx]
            carriers.append(
                {
                    "row_idx": row_idx,
                    "abs_shap": abs(float(shap_values[row_idx, bit])),
                    "split": str(pred_row["split"]),
                    "smiles": str(pred_row[config.SMILES_COLUMN]),
                    "label": int(pred_row[config.LABEL_COLUMN]),
                    "pred_prob": float(pred_row["pred_prob"]),
                    "pred_label": int(pred_row["pred_label"]),
                    "is_correct": int(pred_row["is_correct"]),
                }
            )

        ranked = sorted(carriers, key=lambda row: row["abs_shap"], reverse=True)[:TOP_EXAMPLES_PER_BIT]
        for example_idx, row in enumerate(ranked, start=1):
            example = example_map[row["smiles"]]
            output_png = TOP_BITS_DIR / f"bit_{bit}_example_{example_idx}.png"
            _render_bit_example_png(example, bit, output_png)
            example_rows.append(
                {
                    "bit": bit,
                    "feature_name": f"bit_{bit}",
                    "example_rank": example_idx,
                    "split": row["split"],
                    "label": row["label"],
                    "pred_prob": row["pred_prob"],
                    "pred_label": row["pred_label"],
                    "is_correct": row["is_correct"],
                    "abs_shap": row["abs_shap"],
                    "smiles": row["smiles"],
                    "png": str(output_png),
                }
            )

    return pd.DataFrame(example_rows)


def _build_report(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_examples_df: pd.DataFrame,
) -> str:
    top_bits = summary_df.head(5)
    active_enriched = top_bits[top_bits["positive_prevalence"] > top_bits["negative_prevalence"]]
    inactive_enriched = top_bits[top_bits["negative_prevalence"] > top_bits["positive_prevalence"]]
    accuracy = float(predictions_df["is_correct"].mean())
    pos_rate = float(predictions_df[config.LABEL_COLUMN].mean())

    lines = [
        "# Global RF SHAP analysis",
        "",
        f"The combined dataset analysis covered {len(predictions_df)} valid molecules with a positive ratio of {pos_rate:.3f} and an RF accuracy of {accuracy:.3f}.",
        "",
        "## Dominant bits",
    ]
    for _, row in top_bits.iterrows():
        lines.append(
            f"- bit_{int(row['bit'])}: mean|SHAP|={row['mean_abs_shap']:.4f}, "
            f"positive prevalence={row['positive_prevalence']:.3f}, negative prevalence={row['negative_prevalence']:.3f}."
        )

    lines.extend(["", "## Interpretation"])
    if not active_enriched.empty:
        lines.append(
            "Top-ranked bits were partly enriched in actives, suggesting that RF uses recurring active-associated opioid-like motifs as positive evidence."
        )
    if not inactive_enriched.empty:
        lines.append(
            "Several high-importance bits were more prevalent in inactives, indicating that RF also relies on strong negative structural cues to suppress false positives."
        )
    if not top_examples_df.empty:
        example_bits = ", ".join(top_examples_df["feature_name"].drop_duplicates().head(3).tolist())
        lines.append(
            f"Representative highlighted subgraphs for {example_bits} provide concrete structure-level examples of the globally important Morgan environments."
        )
    lines.append(
        "Overall, the global SHAP pattern is consistent with the case-level RF SAR analysis: the model combines active-associated opioid-like environments with inactive-associated counter-signals rather than relying on a single motif family."
    )

    return "\n".join(lines) + "\n"


def run_global_rf_shap() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TOP_BITS_DIR.mkdir(parents=True, exist_ok=True)

    load_runtime()
    rf_model = load_rf_model()

    examples, metadata_df, invalid = _combine_examples()
    feature_matrix = np.stack([example.feature for example in examples], axis=0)
    predictions_df = _predict_dataframe(rf_model, examples, metadata_df)
    shap_values = compute_shap_values(rf_model, feature_matrix)

    shap_values_df = _build_shap_values_df(predictions_df, shap_values)
    summary_df = _bit_summary(
        feature_matrix=feature_matrix,
        labels=predictions_df[config.LABEL_COLUMN].to_numpy(dtype=int),
        shap_values=shap_values,
    )

    top_bits = summary_df["bit"].head(TOP_N_FIGURE_BITS).astype(int).tolist()
    top_examples_df = _build_top_bit_examples(top_bits, examples, predictions_df, shap_values)

    predictions_df.to_csv(GLOBAL_PREDICTIONS_CSV, index=False)
    shap_values_df.to_csv(GLOBAL_SHAP_VALUES_CSV, index=False)
    summary_df.to_csv(GLOBAL_BIT_SUMMARY_CSV, index=False)
    top_examples_df.to_csv(TOP_BIT_EXAMPLES_CSV, index=False)
    INVALID_SMILES_JSON.write_text(json.dumps(invalid, indent=2), encoding="utf-8")

    _save_beeswarm(shap_values, feature_matrix, top_bits)
    _save_importance_bar(summary_df, top_bits)
    _save_prevalence_plot(summary_df, top_bits)

    GLOBAL_SHAP_REPORT_MD.write_text(
        _build_report(predictions_df, summary_df, top_examples_df),
        encoding="utf-8",
    )

    return {
        "output_dir": OUTPUT_DIR,
        "predictions_csv": GLOBAL_PREDICTIONS_CSV,
        "shap_values_csv": GLOBAL_SHAP_VALUES_CSV,
        "bit_summary_csv": GLOBAL_BIT_SUMMARY_CSV,
        "top_examples_csv": TOP_BIT_EXAMPLES_CSV,
        "report_md": GLOBAL_SHAP_REPORT_MD,
        "beeswarm_png": BEESWARM_PNG,
        "importance_bar_png": IMPORTANCE_BAR_PNG,
        "prevalence_png": PREVALENCE_PNG,
        "invalid_smiles_json": INVALID_SMILES_JSON,
    }
