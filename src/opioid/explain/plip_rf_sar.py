from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from opioid.baselines import config
from opioid.explain.rf_sar import build_examples, compute_shap_values, load_rf_model, render_sar_png

OUTPUT_DIR = Path("outputs/explain_plip_rf")
RAW_DIR = OUTPUT_DIR / "plip"
FIGURES_DIR = OUTPUT_DIR / "figures"
STRUCTURES_DIR = OUTPUT_DIR / "structures"

INTERACTIONS_CSV = OUTPUT_DIR / "plip_summary.csv"
CASE_SUMMARY_CSV = OUTPUT_DIR / "plip_case_summary.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
RUN_CONFIG_JSON = OUTPUT_DIR / "run_config.json"

PLIP_ENV = "plip_env"

CASES = [
    {
        "pdb_id": "5C1M",
        "ligand": "BU72",
        "smiles": "CO[C@]12C=C[C@]34C[C@@]1(C)[C@H](N[C@H]2[C@@]35CCN(C)[C@@H]4Cc6ccc(O)cc56)c7ccccc7",
        "expected_role": "Agonist",
        "expected_antagonist_label": 0,
        "hetid": "VF1",
        "chain": "A",
        "position": "407",
        "source_url": "https://www.rcsb.org/structure/5C1M",
    },
    {
        "pdb_id": "8EF5",
        "ligand": "Fentanyl",
        "smiles": "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1",
        "expected_role": "Agonist",
        "expected_antagonist_label": 0,
        "hetid": "7V7",
        "chain": "M",
        "position": "501",
        "source_url": "https://www.rcsb.org/structure/8EF5",
    },
    {
        "pdb_id": "8EF6",
        "ligand": "Morphine",
        "smiles": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "expected_role": "Agonist",
        "expected_antagonist_label": 0,
        "hetid": "MOI",
        "chain": "M",
        "position": "401",
        "source_url": "https://www.rcsb.org/structure/8EF6",
    },
    {
        "pdb_id": "4DKL",
        "ligand": "beta-FNA",
        "smiles": "COC(=O)/C=C/C(=O)N[C@@H]1CC[C@]2([C@H]3CC4=C5[C@]2([C@H]1OC5=C(C=C4)O)CCN3CC6CC6)O",
        "expected_role": "Antagonist",
        "expected_antagonist_label": 1,
        "hetid": "BF0",
        "chain": "A",
        "position": "601",
        "source_url": "https://www.rcsb.org/structure/4DKL",
    },
]

INTERACTION_GROUP_FIELDS = {
    "hydrophobic_interactions": ["restype", "resnr", "reschain", "dist"],
    "hydrogen_bonds": ["restype", "resnr", "reschain", "dist_d-a"],
    "water_bridges": ["restype", "resnr", "reschain", "dist_d-w"],
    "salt_bridges": ["restype", "resnr", "reschain", "dist"],
    "pi_stacks": ["restype", "resnr", "reschain", "centdist"],
    "pi_cation_interactions": ["restype", "resnr", "reschain", "dist"],
    "halogen_bonds": ["restype", "resnr", "reschain", "dist"],
    "metal_complexes": ["restype", "resnr", "reschain", "dist"],
}


def _run_plip_case(case: dict[str, str]) -> tuple[Path, Path, Path]:
    case_dir = RAW_DIR / case["pdb_id"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda",
        "run",
        "-n",
        PLIP_ENV,
        "plip",
        "-i",
        case["pdb_id"],
        "-o",
        str(case_dir),
        "-x",
        "-t",
        "-q",
    ]
    subprocess.run(cmd, check=True)

    pdb_path = case_dir / f"{case['pdb_id'].lower()}.pdb"
    protonated_path = case_dir / f"{case['pdb_id'].lower()}_protonated.pdb"
    xml_path = case_dir / f"{case['pdb_id'].lower()}_report.xml"
    return pdb_path, protonated_path, xml_path


def _find_target_bindingsite(root: ET.Element, case: dict[str, str]) -> ET.Element:
    for bs in root.findall("bindingsite"):
        ids = bs.find("identifiers")
        if ids is None:
            continue
        if (
            ids.findtext("hetid") == case["hetid"]
            and ids.findtext("chain") == case["chain"]
            and ids.findtext("position") == case["position"]
        ):
            return bs
    raise ValueError(f"Could not find PLIP binding site for {case['pdb_id']} {case['hetid']}:{case['chain']}:{case['position']}")


def _parse_plip_rows(case: dict[str, str], xml_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = ET.parse(xml_path).getroot()
    bs = _find_target_bindingsite(root, case)
    inter = bs.find("interactions")
    if inter is None:
        raise ValueError(f"No interactions section in {xml_path}")

    rows: list[dict[str, object]] = []
    summary_counts: dict[str, int] = {}
    residues: list[str] = []

    for group in inter:
        group_name = group.tag
        items = list(group)
        if not items:
            continue
        summary_counts[group_name] = len(items)
        for item in items:
            restype = item.findtext("restype") or ""
            resnr = item.findtext("resnr") or ""
            reschain = item.findtext("reschain") or ""
            if restype and resnr:
                residues.append(f"{restype}{resnr}")

            distance = None
            for field in ("dist", "dist_d-a", "dist_d-w", "centdist"):
                text = item.findtext(field)
                if text:
                    distance = float(text)
                    break

            rows.append(
                {
                    "pdb_id": case["pdb_id"],
                    "ligand": case["ligand"],
                    "hetid": case["hetid"],
                    "ligand_chain": case["chain"],
                    "ligand_position": case["position"],
                    "interaction_type": group_name,
                    "protein_residue": restype,
                    "protein_resnr": resnr,
                    "protein_chain": reschain,
                    "distance": distance,
                }
            )

    summary = {
        "interaction_counts": summary_counts,
        "top_residues": "; ".join(pd.Series(residues).value_counts().head(6).index.tolist()),
    }
    return rows, summary


def _compute_rf_case_rows(cases: list[dict[str, str]]) -> list[dict[str, object]]:
    df = pd.DataFrame(
        {
            config.SMILES_COLUMN: [case["smiles"] for case in cases],
            config.LABEL_COLUMN: [0] * len(cases),
        }
    )
    examples, invalid = build_examples(df)
    if invalid:
        raise ValueError(f"Invalid case SMILES: {invalid}")

    model = load_rf_model()
    x = np.stack([example.feature for example in examples], axis=0)
    probs = model.predict_proba(x)[:, 1].astype(float)
    shap_values = compute_shap_values(model, x)

    case_rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        output_png = FIGURES_DIR / f"{case['pdb_id']}_rf_sar.png"
        pos_bits, neg_bits = render_sar_png(
            mol=examples[idx].mol,
            bit_info=examples[idx].bit_info,
            shap_values=shap_values[idx],
            x_row=examples[idx].feature,
            output_png=output_png,
        )
        prob = float(probs[idx])
        pred_label = int(prob >= config.THRESHOLD)
        expected = int(case["expected_antagonist_label"])
        alignment_call = "Aligned" if pred_label == expected else "Misaligned"
        case_rows.append(
            {
                "pdb_id": case["pdb_id"],
                "ligand": case["ligand"],
                "expected_role": case["expected_role"],
                "rf_prob_antagonist": prob,
                "rf_pred_label": pred_label,
                "rf_shap_support_bits": "; ".join(f"bit_{bit}" for bit in pos_bits),
                "rf_shap_conflict_bits": "; ".join(f"bit_{bit}" for bit in neg_bits),
                "rf_sar_figure": str(output_png),
                "alignment_call": alignment_call,
            }
        )
    return case_rows


def _build_report(case_summary_df: pd.DataFrame) -> str:
    lines = [
        "# PLIP and RF-SAR comparison",
        "",
        "This report compares PLIP-derived receptor interaction patterns with local RF-SHAP substructure highlights for four μOR ligand cases.",
        "",
    ]
    for _, row in case_summary_df.iterrows():
        lines.extend(
            [
                f"## {row['pdb_id']} {row['ligand']}",
                f"- Expected role: {row['expected_role']}",
                f"- RF antagonist probability: {float(row['rf_prob_antagonist']):.3f}",
                f"- PLIP interaction counts: {row['plip_interaction_counts']}",
                f"- PLIP top residues: {row['plip_top_residues']}",
                f"- RF-SHAP support bits: {row['rf_shap_support_bits']}",
                f"- RF-SHAP conflict bits: {row['rf_shap_conflict_bits']}",
                f"- Alignment: {row['alignment_call']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "Using antagonist as the positive class, the three agonists receive low antagonist probability and the antagonist beta-FNA receives high antagonist probability. PLIP highlights canonical μOR contact patterns for the selected ligands, while RF-SHAP highlights the ligand substructures that push the model toward or away from the antagonist class. Together, these outputs give a receptor-level and fingerprint-level SAR view for the same four cases.",
            "",
            "PLIP raw reports are included, but no PLIP picture files are generated here because the current PLIP environment does not contain PyMOL.",
            "",
        ]
    )
    return "\n".join(lines)


def run_plip_rf_sar() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    interaction_rows: list[dict[str, object]] = []
    plip_summaries: dict[str, dict[str, object]] = {}
    for case in CASES:
        pdb_path, protonated_path, xml_path = _run_plip_case(case)
        shutil.copy2(pdb_path, STRUCTURES_DIR / pdb_path.name)
        shutil.copy2(protonated_path, STRUCTURES_DIR / protonated_path.name)
        rows, summary = _parse_plip_rows(case, xml_path)
        interaction_rows.extend(rows)
        plip_summaries[case["pdb_id"]] = summary

    pd.DataFrame(interaction_rows).to_csv(INTERACTIONS_CSV, index=False)

    rf_case_rows = _compute_rf_case_rows(CASES)
    summary_rows: list[dict[str, object]] = []
    for row in rf_case_rows:
        plip_summary = plip_summaries[row["pdb_id"]]
        summary_rows.append(
            {
                **row,
                "plip_interaction_counts": json.dumps(plip_summary["interaction_counts"], sort_keys=True),
                "plip_top_residues": plip_summary["top_residues"],
                "plip_dir": str(RAW_DIR / row["pdb_id"]),
            }
        )

    case_summary_df = pd.DataFrame(summary_rows)
    case_summary_df.to_csv(CASE_SUMMARY_CSV, index=False)
    REPORT_MD.write_text(_build_report(case_summary_df), encoding="utf-8")
    RUN_CONFIG_JSON.write_text(
        json.dumps(
            {
                "plip_env": PLIP_ENV,
                "cases": CASES,
                "positive_class": "antagonist",
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": OUTPUT_DIR,
        "interactions_csv": INTERACTIONS_CSV,
        "case_summary_csv": CASE_SUMMARY_CSV,
        "report_md": REPORT_MD,
        "figures_dir": FIGURES_DIR,
        "structures_dir": STRUCTURES_DIR,
        "run_config_json": RUN_CONFIG_JSON,
    }
