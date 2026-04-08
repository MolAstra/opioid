from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw

from opioid.baselines import config
from opioid.baselines.data import load_development_set, load_test_split

RDLogger.DisableLog("rdApp.warning")


OUTPUT_DIR = Path("outputs/explain_rf")
ARTIFACTS_DIR = Path("outputs/baselines/artifacts")
RUNTIME_CONFIG = ARTIFACTS_DIR / "runtime_config.json"
RF_MODEL_PATH = ARTIFACTS_DIR / "rf.joblib"
SAR_INDEX_CSV = OUTPUT_DIR / "rf_sar_index.csv"
SAR_SUMMARY_CSV = OUTPUT_DIR / "rf_sar_summary.csv"
SAR_REPORT_MD = OUTPUT_DIR / "rf_sar_report.md"
INVALID_SMILES_JSON = OUTPUT_DIR / "invalid_smiles.json"

TARGET_CASES_PER_SPLIT = 5
TOP_K_BITS = 6
RF_THRESHOLD = 0.5


@dataclass(frozen=True)
class MorganExample:
    smiles: str
    label: int
    mol: Chem.Mol
    feature: np.ndarray
    bit_info: dict[int, list[tuple[int, int]]]


def load_rf_model():
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing RF model at {RF_MODEL_PATH}")
    return joblib.load(RF_MODEL_PATH)


def load_runtime() -> dict:
    return json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))


def build_examples(df: pd.DataFrame) -> tuple[list[MorganExample], list[str]]:
    examples: list[MorganExample] = []
    invalid_smiles: list[str] = []

    for _, row in df.iterrows():
        smiles = str(row[config.SMILES_COLUMN])
        label = int(row[config.LABEL_COLUMN])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append(smiles)
            continue

        bit_info: dict[int, list[tuple[int, int]]] = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=config.MORGAN_RADIUS,
            nBits=config.MORGAN_BITS,
            bitInfo=bit_info,
        )
        arr = np.zeros((config.MORGAN_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        examples.append(
            MorganExample(
                smiles=smiles,
                label=label,
                mol=mol,
                feature=arr,
                bit_info=bit_info,
            )
        )

    return examples, invalid_smiles


def predict_examples(model, examples: list[MorganExample], split: str) -> pd.DataFrame:
    x = np.stack([example.feature for example in examples], axis=0)
    pred_prob = model.predict_proba(x)[:, 1].astype(float)
    pred_label = (pred_prob >= RF_THRESHOLD).astype(int)
    labels = np.asarray([example.label for example in examples], dtype=int)
    is_correct = pred_label == labels

    df = pd.DataFrame(
        {
            "split": split,
            "smiles": [example.smiles for example in examples],
            "label": labels,
            "pred_prob": pred_prob,
            "pred_label": pred_label,
            "is_correct": is_correct.astype(int),
        }
    )
    df["margin"] = np.abs(df["pred_prob"] - RF_THRESHOLD)
    df["case_type"] = np.where(df["is_correct"] == 1, "success", "failure")
    df["error_type"] = "correct"
    df.loc[(df["label"] == 0) & (df["pred_label"] == 1), "error_type"] = "false_positive"
    df.loc[(df["label"] == 1) & (df["pred_label"] == 0), "error_type"] = "false_negative"
    return df


def _pick_diverse_subset(df: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    used_indices: set[int] = set()

    success_df = df[df["case_type"] == "success"].sort_values("margin", ascending=False)
    failure_df = df[df["case_type"] == "failure"].sort_values("margin", ascending=False)

    preferred_groups = [
        success_df[success_df["label"] == 1],
        success_df[success_df["label"] == 0],
        failure_df[failure_df["error_type"] == "false_positive"],
        failure_df[failure_df["error_type"] == "false_negative"],
    ]

    for group in preferred_groups:
        for _, row in group.iterrows():
            idx = int(row.name)
            if idx in used_indices:
                continue
            selected_rows.append(row)
            used_indices.add(idx)
            break

    target_successes = min(3, int((df["case_type"] == "success").sum()))
    target_failures = min(2, int((df["case_type"] == "failure").sum()))

    while sum(row["case_type"] == "success" for row in selected_rows) < target_successes:
        for _, row in success_df.iterrows():
            idx = int(row.name)
            if idx not in used_indices:
                selected_rows.append(row)
                used_indices.add(idx)
                break
        else:
            break

    while sum(row["case_type"] == "failure" for row in selected_rows) < target_failures:
        for _, row in failure_df.iterrows():
            idx = int(row.name)
            if idx not in used_indices:
                selected_rows.append(row)
                used_indices.add(idx)
                break
        else:
            break

    ranked_all = df.sort_values(
        by=["case_type", "margin"],
        ascending=[True, False],
        key=lambda col: col.map({"success": 0, "failure": 1}) if col.name == "case_type" else col,
    )
    for _, row in ranked_all.iterrows():
        idx = int(row.name)
        if idx in used_indices:
            continue
        selected_rows.append(row)
        used_indices.add(idx)
        if len(selected_rows) >= n_cases:
            break

    selected = pd.DataFrame(selected_rows).head(n_cases).copy()
    return selected.reset_index(drop=True)


def select_cases(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_selected = _pick_diverse_subset(train_df, TARGET_CASES_PER_SPLIT)
    test_selected = _pick_diverse_subset(test_df, TARGET_CASES_PER_SPLIT)
    return pd.concat([train_selected, test_selected], ignore_index=True)


def compute_shap_values(model, feature_matrix: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(feature_matrix)
    if isinstance(values, list):
        arr = values[1] if len(values) > 1 else values[0]
    else:
        arr = values
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def collect_bit_atoms_bonds(
    mol: Chem.Mol,
    bit_info: dict[int, list[tuple[int, int]]],
    bits: list[int],
) -> tuple[set[int], set[int]]:
    atoms: set[int] = set()
    bonds: set[int] = set()

    for bit in bits:
        for center, radius in bit_info.get(int(bit), []):
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
            if env_bonds:
                for bond_idx in env_bonds:
                    bonds.add(int(bond_idx))
                    bond = mol.GetBondWithIdx(int(bond_idx))
                    atoms.add(int(bond.GetBeginAtomIdx()))
                    atoms.add(int(bond.GetEndAtomIdx()))
            else:
                atoms.add(int(center))

    return atoms, bonds


def render_sar_png(
    mol: Chem.Mol,
    bit_info: dict[int, list[tuple[int, int]]],
    shap_values: np.ndarray,
    x_row: np.ndarray,
    output_png: Path,
) -> tuple[list[int], list[int]]:
    on_bits = np.where(x_row > 0.5)[0]
    bit_values = [(int(bit), float(shap_values[int(bit)])) for bit in on_bits]
    pos_bits = [bit for bit, _ in sorted((pair for pair in bit_values if pair[1] > 0), key=lambda item: item[1], reverse=True)[:TOP_K_BITS]]
    neg_bits = [bit for bit, _ in sorted((pair for pair in bit_values if pair[1] < 0), key=lambda item: item[1])[:TOP_K_BITS]]

    pos_atoms, pos_bonds = collect_bit_atoms_bonds(mol, bit_info, pos_bits)
    neg_atoms, neg_bonds = collect_bit_atoms_bonds(mol, bit_info, neg_bits)

    atom_colors: dict[int, tuple[float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float]] = {}

    for atom_idx in pos_atoms:
        atom_colors[atom_idx] = (0.20, 0.72, 0.33)
    for atom_idx in neg_atoms:
        atom_colors[atom_idx] = (0.84, 0.22, 0.18) if atom_idx not in atom_colors else (0.92, 0.58, 0.18)

    for bond_idx in pos_bonds:
        bond_colors[bond_idx] = (0.20, 0.72, 0.33)
    for bond_idx in neg_bonds:
        bond_colors[bond_idx] = (0.84, 0.22, 0.18) if bond_idx not in bond_colors else (0.92, 0.58, 0.18)

    image = Draw.MolToImage(
        mol,
        size=(900, 520),
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
    return pos_bits, neg_bits


def case_note(row: pd.Series) -> str:
    if row["case_type"] == "success" and row["label"] == 1:
        return "successful active prediction"
    if row["case_type"] == "success" and row["label"] == 0:
        return "successful inactive prediction"
    if row["error_type"] == "false_positive":
        return "false positive"
    if row["error_type"] == "false_negative":
        return "false negative"
    return "prediction case"


def interpret_case(row: pd.Series, pos_bits: list[int], neg_bits: list[int]) -> str:
    pos_txt = ", ".join(f"bit_{bit}" for bit in pos_bits[:3]) or "no strong positive bits"
    neg_txt = ", ".join(f"bit_{bit}" for bit in neg_bits[:3]) or "no strong negative bits"
    if row["case_type"] == "success":
        return f"Supportive subgraphs driven by {pos_txt} outweighed opposing motifs from {neg_txt}."
    return f"The RF decision was likely misled by supportive motifs from {pos_txt} relative to opposing cues from {neg_txt}."


def build_report(summary_df: pd.DataFrame) -> str:
    success_rows = summary_df[summary_df["case_type"] == "success"].head(3)
    failure_rows = summary_df[summary_df["case_type"] == "failure"].head(3)

    lines = [
        "# RF SAR analysis",
        "",
        "## Success cases",
    ]
    for _, row in success_rows.iterrows():
        lines.append(f"- {row['display_id']} ({row['split']}): {row['interpretation']}")

    lines.extend(["", "## Failure cases"])
    for _, row in failure_rows.iterrows():
        lines.append(f"- {row['display_id']} ({row['split']}): {row['interpretation']}")

    return "\n".join(lines) + "\n"


def run_rf_sar() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_runtime()
    rf_model = load_rf_model()

    train_examples, train_invalid = build_examples(load_development_set())
    test_examples, test_invalid = build_examples(load_test_split())

    train_pred_df = predict_examples(rf_model, train_examples, split="train_valid")
    test_pred_df = predict_examples(rf_model, test_examples, split="test")
    selected_df = select_cases(train_pred_df, test_pred_df)

    all_examples = {
        ("train_valid", example.smiles): example for example in train_examples
    } | {
        ("test", example.smiles): example for example in test_examples
    }

    selected_features = np.stack(
        [
            all_examples[(str(row["split"]), str(row["smiles"]))].feature
            for _, row in selected_df.iterrows()
        ],
        axis=0,
    )
    shap_values = compute_shap_values(rf_model, selected_features)

    index_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(selected_df.iterrows(), start=1):
        example = all_examples[(str(row["split"]), str(row["smiles"]))]
        display_id = f"RF-{idx:02d}"
        output_png = OUTPUT_DIR / f"rf_{row['split']}_{row['case_type']}_{idx:02d}.png"
        pos_bits, neg_bits = render_sar_png(
            mol=example.mol,
            bit_info=example.bit_info,
            shap_values=shap_values[idx - 1],
            x_row=example.feature,
            output_png=output_png,
        )
        interpretation = interpret_case(row, pos_bits, neg_bits)

        index_rows.append(
            {
                "display_id": display_id,
                "split": row["split"],
                "case_type": row["case_type"],
                "error_type": row["error_type"],
                "label": int(row["label"]),
                "pred_label": int(row["pred_label"]),
                "pred_prob": float(row["pred_prob"]),
                "is_correct": int(row["is_correct"]),
                "smiles": row["smiles"],
                "png": str(output_png),
                "top_pos_bits": ";".join(str(bit) for bit in pos_bits),
                "top_neg_bits": ";".join(str(bit) for bit in neg_bits),
            }
        )
        summary_rows.append(
            {
                "display_id": display_id,
                "split": row["split"],
                "case_type": row["case_type"],
                "label": int(row["label"]),
                "pred_prob": float(row["pred_prob"]),
                "case_note": case_note(row),
                "interpretation": interpretation,
            }
        )

    pd.DataFrame(index_rows).to_csv(SAR_INDEX_CSV, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SAR_SUMMARY_CSV, index=False)
    SAR_REPORT_MD.write_text(build_report(summary_df), encoding="utf-8")
    INVALID_SMILES_JSON.write_text(
        json.dumps(
            {
                "train_valid": train_invalid,
                "test": test_invalid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "sar_index_csv": SAR_INDEX_CSV,
        "sar_summary_csv": SAR_SUMMARY_CSV,
        "sar_report_md": SAR_REPORT_MD,
        "output_dir": OUTPUT_DIR,
        "train_invalid_json": INVALID_SMILES_JSON,
    }
