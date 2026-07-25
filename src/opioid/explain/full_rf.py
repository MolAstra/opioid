"""Full-corpus RF/ECFP4 explanation and screening-model bundle."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import shap

from opioid.benchmark.catalog import CandidateSpec, traditional_candidates
from opioid.benchmark.display_names import get_model_display
from opioid.benchmark.estimators import DEFAULT_SEED, build_estimator
from opioid.chemistry.representations import (
    MORGAN_RADIUS,
    FeatureDataset,
    build_feature_dataset,
)


FULL_RF_SCHEMA_VERSION = 2
MODEL_CANDIDATE_ID = "rf__ecfp_2048"
EXPECTED_ROWS = 982
EXPECTED_CLASS_COUNTS = {0: 754, 1: 228}
PROTOCOLS = ("scaffold", "random", "datasail")
TOP_FIGURE_BITS = 20
TOP_REPORT_BITS = 12
LOCAL_CASES_PER_CATEGORY = 4
LOCAL_CATEGORIES = ("high_antagonistic", "high_non_antagonistic", "boundary")


@dataclass(frozen=True)
class FullCorpus:
    """Canonical full labeled corpus reconstructed from every formal partition."""

    frame: pd.DataFrame
    input_files: dict[str, str]


@dataclass(frozen=True)
class ExplanationBundle:
    """Paths written by one complete explanation run."""

    root: Path
    model_path: Path
    model_manifest_path: Path
    report_path: Path
    manifest_path: Path


def explanation_candidate() -> CandidateSpec:
    """Return the fixed user-approved RF/ECFP4-2048 explanation target."""

    return next(
        candidate
        for candidate in traditional_candidates()
        if candidate.candidate_id == MODEL_CANDIDATE_ID
    )


def load_full_corpus(task_root: Path | str) -> FullCorpus:
    """Reconstruct and cross-check the identical 982-row corpus in all protocols."""

    root = Path(task_root)
    protocol_frames: dict[str, pd.DataFrame] = {}
    input_files: dict[str, str] = {}
    for protocol in PROTOCOLS:
        split_root = root / f"processed_{protocol}"
        paths = [
            split_root / "train_fold-0.csv",
            split_root / "valid_fold-0.csv",
            split_root / "test.csv",
        ]
        manifest_path = split_root / "manifest.json"
        for path in [*paths, manifest_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing formal full-corpus input: {path}")
            input_files[str(path.relative_to(root))] = _sha256(path)

        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        required = {"sample_id", "smiles", "label"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{protocol} full corpus is missing columns: {missing}")
        frame = frame[["sample_id", "smiles", "label"]].copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["smiles"] = frame["smiles"].astype(str)
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
        duplicate_conflicts = frame.groupby("sample_id")[["smiles", "label"]].nunique()
        if (duplicate_conflicts > 1).any(axis=None):
            raise ValueError(f"{protocol} contains conflicting duplicate sample IDs")
        frame = frame.drop_duplicates("sample_id")
        frame["_sample_order"] = pd.to_numeric(frame["sample_id"], errors="raise")
        protocol_frames[protocol] = (
            frame.sort_values("_sample_order", kind="stable")
            .drop(columns="_sample_order")
            .reset_index(drop=True)
        )

    reference = protocol_frames[PROTOCOLS[0]]
    for protocol in PROTOCOLS[1:]:
        if not protocol_frames[protocol].equals(reference):
            raise ValueError(f"{protocol} does not contain the same canonical labeled corpus")
    counts = reference["label"].value_counts().sort_index().to_dict()
    if len(reference) != EXPECTED_ROWS or counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            f"Full corpus contract changed: rows={len(reference)}, class_counts={counts}"
        )
    return FullCorpus(frame=reference, input_files=input_files)


def train_full_rf(corpus: FullCorpus) -> tuple[Any, FeatureDataset]:
    """Fit the fixed RF target on all deduplicated labeled rows."""

    candidate = explanation_candidate()
    dataset = build_feature_dataset(corpus.frame, candidate.feature_spec)
    if dataset.n_invalid:
        raise ValueError(f"Full corpus contains invalid SMILES: {dataset.invalid_smiles[:5]}")
    if dataset.labels is None or dataset.n_rows != EXPECTED_ROWS:
        raise ValueError("Full-corpus featurization did not preserve all labels and rows")
    model = build_estimator(candidate, seed=DEFAULT_SEED)
    model.fit(dataset.features, dataset.labels)
    return model, dataset


def compute_class1_shap(model: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    """Return class-1 TreeSHAP values and verify probability additivity."""

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(features)
    values = _select_class1_values(raw_values)
    expected_value = _select_class1_expected_value(explainer.expected_value)
    probabilities = model.predict_proba(features)[:, 1]
    reconstructed = expected_value + values.sum(axis=1)
    if not np.allclose(reconstructed, probabilities, rtol=1e-5, atol=1e-5):
        error = float(np.max(np.abs(reconstructed - probabilities)))
        raise ValueError(f"TreeSHAP class-1 additivity failed; max error={error:.3g}")
    return np.asarray(values, dtype=np.float32), expected_value


def write_explanation_package(
    *,
    task_root: Path | str,
    benchmark_root: Path | str,
    report_root: Path | str,
    destination: Path | str,
) -> ExplanationBundle:
    """Train the model and write a self-contained explanation/screening bundle."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    figures_dir = output / "figures"
    environments_dir = figures_dir / "bit_environments"
    local_dir = figures_dir / "local_cases"
    for path in (model_dir, figures_dir, environments_dir, local_dir):
        path.mkdir(parents=True, exist_ok=True)

    corpus = load_full_corpus(task_root)
    selection = _load_selection_evidence(Path(report_root))
    model, dataset = train_full_rf(corpus)
    probabilities = model.predict_proba(dataset.features)[:, 1].astype(float)
    shap_values, expected_value = compute_class1_shap(model, dataset.features)

    model_path = model_dir / f"{MODEL_CANDIDATE_ID}.joblib"
    joblib.dump(model, model_path)
    sample_index = _sample_index(dataset, probabilities)
    sample_index_path = output / "sample_index.csv"
    sample_index.to_csv(sample_index_path, index=False)
    np.savez_compressed(
        output / "shap_values.npz",
        values=shap_values,
        expected_value=np.asarray([expected_value], dtype=np.float64),
    )

    bit_summary = _bit_summary(dataset, shap_values)
    bit_summary_path = output / "bit_summary.csv"
    bit_summary.to_csv(bit_summary_path, index=False)
    top_bits = bit_summary.head(TOP_FIGURE_BITS)["bit"].astype(int).tolist()
    environments, examples = _bit_environments(dataset, shap_values, top_bits)
    environments_path = output / "bit_environments.csv"
    environments.to_csv(environments_path, index=False)
    _render_environment_examples(examples, environments_dir)

    local_cases = _select_local_cases(dataset, probabilities)
    local_cases = _render_local_cases(
        dataset,
        shap_values,
        local_cases,
        output,
        local_dir,
    )
    local_cases_path = output / "local_cases.csv"
    local_cases.to_csv(local_cases_path, index=False)

    _plot_global_importance(bit_summary, figures_dir / "global_importance.png")
    _plot_class_prevalence(bit_summary, figures_dir / "class_prevalence.png")
    _plot_beeswarm(
        shap_values,
        dataset.features,
        top_bits,
        figures_dir / "global_beeswarm.png",
    )

    model_manifest = _model_manifest(
        corpus=corpus,
        dataset=dataset,
        model_path=model_path,
        selection=selection,
        expected_value=expected_value,
        benchmark_root=Path(benchmark_root),
        report_root=Path(report_root),
    )
    model_manifest_path = model_dir / "model_manifest.json"
    model_manifest_path.write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output / "report.md"
    report_path.write_text(
        _build_report(bit_summary, environments, local_cases, selection),
        encoding="utf-8",
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(_package_manifest(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExplanationBundle(
        root=output,
        model_path=model_path,
        model_manifest_path=model_manifest_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )


def load_full_rf_bundle(root: Path | str) -> tuple[Any, dict[str, Any]]:
    """Load and validate the full-data RF bundle for explanation or screening."""

    bundle_root = Path(root)
    manifest_path = bundle_root / "model" / "model_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != FULL_RF_SCHEMA_VERSION
        or payload.get("candidate", {}).get("candidate_id") != MODEL_CANDIDATE_ID
        or payload.get("training_scope") != "full_deduplicated_labeled_corpus"
        or payload.get("independent_evaluation") is not False
    ):
        raise ValueError("Model manifest does not describe the approved full-data RF bundle")
    model_path = bundle_root / str(payload["artifact_path"])
    if _sha256(model_path) != payload.get("artifact_sha256"):
        raise ValueError("Full-data RF artifact checksum does not match its manifest")
    return joblib.load(model_path), payload


def _sample_index(dataset: FeatureDataset, probabilities: np.ndarray) -> pd.DataFrame:
    frame = dataset.rows[["sample_id", "smiles", "label"]].copy()
    frame["canonical_smiles"] = dataset.canonical_smiles
    frame["antagonistic_score"] = probabilities
    frame["training_member"] = True
    return frame


def _bit_summary(dataset: FeatureDataset, shap_values: np.ndarray) -> pd.DataFrame:
    if dataset.labels is None:
        raise ValueError("Bit summaries require labels")
    labels = dataset.labels
    positive = labels == 1
    negative = labels == 0
    features = dataset.features
    frame = pd.DataFrame(
        {
            "bit": np.arange(features.shape[1], dtype=int),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "carrier_count": features.sum(axis=0).astype(int),
            "antagonistic_prevalence": features[positive].mean(axis=0),
            "non_antagonistic_prevalence": features[negative].mean(axis=0),
        }
    )
    return frame.sort_values(
        ["mean_abs_shap", "carrier_count", "bit"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _morgan_example(canonical_smiles: str) -> tuple[Chem.Mol, dict[int, list[tuple[int, int]]]]:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Could not parse canonical SMILES: {canonical_smiles}")
    bit_info: dict[int, list[tuple[int, int]]] = {}
    AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=MORGAN_RADIUS,
        nBits=2048,
        bitInfo=bit_info,
    )
    return molecule, bit_info


def _environment_smiles(molecule: Chem.Mol, center: int, radius: int) -> str:
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center))
    atoms = {center}
    for bond_index in bonds:
        bond = molecule.GetBondWithIdx(int(bond_index))
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return Chem.MolFragmentToSmiles(
        molecule,
        atomsToUse=sorted(atoms),
        bondsToUse=sorted(bonds),
        rootedAtAtom=center,
        canonical=True,
        isomericSmiles=True,
    )


def _bit_environments(
    dataset: FeatureDataset,
    shap_values: np.ndarray,
    top_bits: list[int],
) -> tuple[pd.DataFrame, dict[int, tuple[Chem.Mol, dict[int, list[tuple[int, int]]]]]]:
    top_set = set(top_bits)
    counts: Counter[tuple[int, str, int]] = Counter()
    molecule_ids: defaultdict[tuple[int, str, int], set[str]] = defaultdict(set)
    shap_by_environment: defaultdict[tuple[int, str, int], list[float]] = defaultdict(list)
    examples: dict[int, tuple[Chem.Mol, dict[int, list[tuple[int, int]]]]] = {}
    example_strength: dict[int, float] = {}
    for row_index, canonical in enumerate(dataset.canonical_smiles):
        molecule, bit_info = _morgan_example(canonical)
        sample_id = str(dataset.rows.iloc[row_index]["sample_id"])
        for bit in top_set.intersection(bit_info):
            strength = abs(float(shap_values[row_index, bit]))
            if strength > example_strength.get(bit, -1.0):
                example_strength[bit] = strength
                examples[bit] = (molecule, bit_info)
            seen_keys: set[tuple[int, str, int]] = set()
            for center, radius in bit_info[bit]:
                environment = _environment_smiles(molecule, int(center), int(radius))
                key = (bit, environment, int(radius))
                counts[key] += 1
                molecule_ids[key].add(sample_id)
                if key not in seen_keys:
                    shap_by_environment[key].append(float(shap_values[row_index, bit]))
                    seen_keys.add(key)
    rank = {bit: index + 1 for index, bit in enumerate(top_bits)}
    rows = [
        {
            "bit_rank": rank[bit],
            "bit": bit,
            "environment_smiles": environment,
            "radius": radius,
            "occurrence_count": count,
            "molecule_count": len(molecule_ids[(bit, environment, radius)]),
            "mean_carrier_shap": float(np.mean(shap_by_environment[(bit, environment, radius)])),
        }
        for (bit, environment, radius), count in counts.items()
    ]
    frame = pd.DataFrame(rows).sort_values(
        ["bit_rank", "molecule_count", "occurrence_count", "environment_smiles"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    return frame.reset_index(drop=True), examples


def _render_environment_examples(
    examples: dict[int, tuple[Chem.Mol, dict[int, list[tuple[int, int]]]]],
    destination: Path,
) -> None:
    for bit, (molecule, bit_info) in examples.items():
        atoms, bonds = _atoms_and_bonds(molecule, bit_info, [bit])
        image = _draw_highlighted_molecule(
            molecule,
            atom_colors={atom: (0.18, 0.51, 0.77) for atom in atoms},
            bond_colors={bond: (0.18, 0.51, 0.77) for bond in bonds},
        )
        _save_pil(image, destination / f"bit_{bit}.png")


def _select_local_cases(dataset: FeatureDataset, probabilities: np.ndarray) -> pd.DataFrame:
    if dataset.labels is None:
        raise ValueError("Local cases require labels")
    frame = dataset.rows[["sample_id", "smiles", "label"]].copy()
    frame["row_index"] = np.arange(len(frame), dtype=int)
    frame["antagonistic_score"] = probabilities
    definitions = {
        "high_antagonistic": frame.loc[frame["label"] == 1].sort_values(
            ["antagonistic_score", "sample_id"], ascending=[False, True], kind="stable"
        ),
        "high_non_antagonistic": frame.loc[frame["label"] == 0].sort_values(
            ["antagonistic_score", "sample_id"], ascending=[True, True], kind="stable"
        ),
        "boundary": frame.assign(
            boundary_distance=(frame["antagonistic_score"] - 0.5).abs()
        ).sort_values(["boundary_distance", "sample_id"], kind="stable"),
    }
    selected: list[pd.DataFrame] = []
    excluded: set[int] = set()
    for category in LOCAL_CATEGORIES:
        pool = (
            definitions[category]
            .loc[~definitions[category]["row_index"].isin(excluded)]
            .head(LOCAL_CASES_PER_CATEGORY * 5)
        )
        picked = _diverse_pick(pool, dataset.features, LOCAL_CASES_PER_CATEGORY)
        picked.insert(0, "category", category)
        picked.insert(1, "category_rank", np.arange(1, len(picked) + 1, dtype=int))
        selected.append(picked)
        excluded.update(picked["row_index"].astype(int))
    return pd.concat(selected, ignore_index=True)


def _diverse_pick(pool: pd.DataFrame, features: np.ndarray, count: int) -> pd.DataFrame:
    if len(pool) < count:
        raise ValueError(f"Local case pool has only {len(pool)} rows; requires {count}")
    positions = pool["row_index"].astype(int).tolist()
    selected = [positions[0]]
    while len(selected) < count:
        best_position = None
        best_distance = -1.0
        for position in positions:
            if position in selected:
                continue
            similarity = max(_tanimoto(features[position], features[item]) for item in selected)
            distance = 1.0 - similarity
            if distance > best_distance:
                best_distance = distance
                best_position = position
        if best_position is None:
            break
        selected.append(best_position)
    return (
        pool.set_index("row_index", drop=False)
        .loc[selected]
        .drop(columns=["boundary_distance"], errors="ignore")
        .reset_index(drop=True)
    )


def _tanimoto(left: np.ndarray, right: np.ndarray) -> float:
    intersection = float(np.logical_and(left > 0, right > 0).sum())
    union = float(np.logical_or(left > 0, right > 0).sum())
    return intersection / union if union else 1.0


def _render_local_cases(
    dataset: FeatureDataset,
    shap_values: np.ndarray,
    cases: pd.DataFrame,
    output_root: Path,
    destination: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in cases.to_dict(orient="records"):
        index = int(row["row_index"])
        molecule, bit_info = _morgan_example(dataset.canonical_smiles[index])
        active_bits = np.flatnonzero(dataset.features[index] > 0)
        positive = sorted(active_bits, key=lambda bit: shap_values[index, bit], reverse=True)[:4]
        negative = sorted(active_bits, key=lambda bit: shap_values[index, bit])[:4]
        positive = [int(bit) for bit in positive if shap_values[index, bit] > 0]
        negative = [int(bit) for bit in negative if shap_values[index, bit] < 0]
        positive_atoms, positive_bonds = _atoms_and_bonds(molecule, bit_info, positive)
        negative_atoms, negative_bonds = _atoms_and_bonds(molecule, bit_info, negative)
        atom_colors = {
            atom: _contribution_color(atom, positive_atoms, negative_atoms)
            for atom in positive_atoms | negative_atoms
        }
        bond_colors = {
            bond: _contribution_color(bond, positive_bonds, negative_bonds)
            for bond in positive_bonds | negative_bonds
        }
        filename = f"{row['category']}_{int(row['category_rank']):02d}.png"
        image = _draw_highlighted_molecule(
            molecule,
            atom_colors=atom_colors,
            bond_colors=bond_colors,
        )
        output_path = destination / filename
        _save_pil(image, output_path)
        rows.append(
            {
                **row,
                "top_positive_bits": ";".join(map(str, positive)),
                "top_negative_bits": ";".join(map(str, negative)),
                "figure": str(output_path.relative_to(output_root)),
                "training_member": True,
            }
        )
    return pd.DataFrame(rows).drop(columns="row_index")


def _atoms_and_bonds(
    molecule: Chem.Mol,
    bit_info: dict[int, list[tuple[int, int]]],
    bits: list[int],
) -> tuple[set[int], set[int]]:
    atoms: set[int] = set()
    bonds: set[int] = set()
    for bit in bits:
        for center, radius in bit_info.get(int(bit), []):
            environment = Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center)
            if not environment:
                atoms.add(int(center))
            for bond_index in environment:
                bonds.add(int(bond_index))
                bond = molecule.GetBondWithIdx(int(bond_index))
                atoms.update((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
    return atoms, bonds


def _plot_global_importance(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary.head(TOP_FIGURE_BITS).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.barh([f"bit_{bit}" for bit in frame["bit"]], frame["mean_abs_shap"], color="#356D9A")
    ax.set_xlabel("Mean |SHAP value| for class 1")
    ax.set_ylabel("Morgan bit")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_class_prevalence(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary.head(TOP_REPORT_BITS)
    positions = np.arange(len(frame))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.bar(
        positions - width / 2,
        frame["antagonistic_prevalence"],
        width,
        label="Antagonistic (label=1)",
        color="#C15A2E",
    )
    ax.bar(
        positions + width / 2,
        frame["non_antagonistic_prevalence"],
        width,
        label="Non-antagonistic (label=0)",
        color="#4D7F4E",
    )
    ax.set_xticks(positions, [f"bit_{bit}" for bit in frame["bit"]], rotation=45, ha="right")
    ax.set_ylabel("Corpus prevalence")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_beeswarm(
    values: np.ndarray,
    features: np.ndarray,
    top_bits: list[int],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 7.0))
    shap.summary_plot(
        values[:, top_bits],
        features[:, top_bits],
        feature_names=[f"bit_{bit}" for bit in top_bits],
        show=False,
        plot_size=None,
    )
    plt.xlabel("SHAP value for antagonistic class (label=1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _load_selection_evidence(report_root: Path) -> dict[str, Any]:
    path = report_root / "all_candidates.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Workflow 40 candidate evidence: {path}")
    frame = pd.read_csv(path)
    required = {"protocol", "candidate_id", "valid_roc_mean"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Workflow 40 candidate evidence is missing columns: {missing}")
    aggregate = frame.groupby("candidate_id", as_index=False)["valid_roc_mean"].mean()
    aggregate = aggregate.sort_values(
        ["valid_roc_mean", "candidate_id"], ascending=[False, True], kind="stable"
    )
    selected = aggregate.loc[aggregate["candidate_id"] == MODEL_CANDIDATE_ID]
    if selected.empty or aggregate.iloc[0]["candidate_id"] != MODEL_CANDIDATE_ID:
        raise ValueError(
            "RF/ECFP4-2048 is no longer the leading three-protocol mean validation AUROC"
        )
    return {
        "candidate_id": MODEL_CANDIDATE_ID,
        "rule": "highest mean validation AUROC across scaffold, random, and DataSAIL",
        "mean_validation_auroc": float(selected.iloc[0]["valid_roc_mean"]),
        "evidence_sha256": _sha256(path),
    }


def _model_manifest(
    *,
    corpus: FullCorpus,
    dataset: FeatureDataset,
    model_path: Path,
    selection: dict[str, Any],
    expected_value: float,
    benchmark_root: Path,
    report_root: Path,
) -> dict[str, Any]:
    candidate = explanation_candidate()
    display = get_model_display(candidate.candidate_id)
    return {
        "schema_version": FULL_RF_SCHEMA_VERSION,
        "purpose": "full_corpus_explanation_and_screening",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            **candidate.manifest(),
            "model_name": display.model_name,
            "model_type": display.model_type,
        },
        "training_scope": "full_deduplicated_labeled_corpus",
        "independent_evaluation": False,
        "performance_evidence": "workflow/30_benchmark_models and workflow/40_report_results",
        "selection": selection,
        "intended_uses": ["explanation", "screening"],
        "score_semantics": "uncalibrated class-1 random-forest score",
        "label_contract": {"1": "antagonistic", "0": "non-antagonistic"},
        "training_rows": dataset.n_rows,
        "class_counts": {str(key): value for key, value in EXPECTED_CLASS_COUNTS.items()},
        "seed": DEFAULT_SEED,
        "tree_shap_expected_value_class_1": expected_value,
        "artifact_path": f"model/{model_path.name}",
        "artifact_sha256": _sha256(model_path),
        "artifact_published": False,
        "artifact_regeneration_command": "./workflow/50_explain_rf/run.sh all --replace",
        "input_files": corpus.input_files,
        "upstream": {
            "benchmark_validation_sha256": _optional_sha256(
                benchmark_root / "validation_summary.json"
            ),
            "report_manifest_sha256": _optional_sha256(report_root / "manifest.json"),
        },
        "environment": {
            "python_packages": {
                name: _package_version(name)
                for name in ("joblib", "numpy", "pandas", "scikit-learn", "shap")
            },
            "rdkit": rdBase.rdkitVersion,
        },
    }


def _build_report(
    summary: pd.DataFrame,
    environments: pd.DataFrame,
    local_cases: pd.DataFrame,
    selection: dict[str, Any],
) -> str:
    top = summary.head(TOP_REPORT_BITS)
    table = "\n".join(
        [
            "| Rank | Bit | Mean \\|SHAP\\| | Mean SHAP | Label-1 prevalence | Label-0 prevalence |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {rank} | bit_{int(row.bit)} | {row.mean_abs_shap:.4f} | "
                f"{row.mean_shap:.4f} | {row.antagonistic_prevalence:.3f} | "
                f"{row.non_antagonistic_prevalence:.3f} |"
                for rank, row in enumerate(top.itertuples(index=False), start=1)
            ],
        ]
    )
    collision_bits = int(environments.groupby("bit")["environment_smiles"].nunique().gt(1).sum())
    case_links = "\n".join(
        f"- `{row.category}` #{int(row.category_rank)}: label={int(row.label)}, "
        f"score={row.antagonistic_score:.3f}, [{row.sample_id}]({row.figure})"
        for row in local_cases.itertuples(index=False)
    )
    return "\n".join(
        [
            "# RF/ECFP4-2048 full-corpus explanation",
            "",
            "## Contract",
            "",
            f"The model was fixed from validation evidence only: {selection['rule']} "
            f"({selection['mean_validation_auroc']:.4f}). It was then refit on all 982 "
            "deduplicated labeled molecules. This refit has no independent holdout metric; "
            "Workflow 30 remains the performance evidence.",
            "",
            "`label=1` means antagonistic. Scores are uncalibrated class-1 RF scores and are "
            "suitable for relative screening priority, not calibrated probability claims.",
            "",
            "## Global class-1 TreeSHAP",
            "",
            "![Global importance](figures/global_importance.png)",
            "",
            "![SHAP beeswarm](figures/global_beeswarm.png)",
            "",
            "![Class prevalence](figures/class_prevalence.png)",
            "",
            table,
            "",
            "## Morgan environments",
            "",
            f"Among the top {TOP_FIGURE_BITS} hashed bits, {collision_bits} map to more than one "
            "observed atom environment. `bit_environments.csv` retains every observed mapping "
            "instead of assigning a single pharmacophore name to a collided bit.",
            "",
            "## Training-corpus behavior cases",
            "",
            "These examples are selected from molecules used for the full-data refit. They show "
            "model behavior and are not TP/FP or generalization-error evidence. Green environments "
            "increase the class-1 score; red environments decrease it.",
            "",
            case_links,
            "",
        ]
    )


def _package_manifest(root: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".joblib"
    }
    runtime_artifacts = {
        str(path.relative_to(root)): {
            "sha256": _sha256(path),
            "published": False,
        }
        for path in sorted(root.rglob("*.joblib"))
    }
    return {
        "schema_version": FULL_RF_SCHEMA_VERSION,
        "kind": "muorscreen.full_rf_explanation_package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": MODEL_CANDIDATE_ID,
        "training_rows": EXPECTED_ROWS,
        "files": files,
        "runtime_artifacts": runtime_artifacts,
    }


def _select_class1_values(raw_values: Any) -> np.ndarray:
    if isinstance(raw_values, list):
        values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    else:
        values = np.asarray(raw_values)
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim != 2 or values.shape[1] != 2048:
        raise ValueError(f"Unexpected TreeSHAP value shape: {values.shape}")
    return values


def _select_class1_expected_value(raw_value: Any) -> float:
    values = np.asarray(raw_value).reshape(-1)
    return float(values[1] if values.size > 1 else values[0])


def _save_pil(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    Image.open(buffer).save(output_path)


def _draw_highlighted_molecule(
    molecule: Chem.Mol,
    *,
    atom_colors: dict[int, tuple[float, float, float]],
    bond_colors: dict[int, tuple[float, float, float]],
) -> Image.Image:
    """Render custom contribution colors with RDKit's explicit Cairo API."""

    prepared = rdMolDraw2D.PrepareMolForDrawing(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(900, 520)
    drawer.DrawMolecule(
        prepared,
        highlightAtoms=sorted(atom_colors),
        highlightBonds=sorted(bond_colors),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def _contribution_color(
    item: int,
    positive: set[int],
    negative: set[int],
) -> tuple[float, float, float]:
    if item in positive and item in negative:
        return (0.92, 0.58, 0.18)
    if item in positive:
        return (0.20, 0.72, 0.33)
    return (0.84, 0.22, 0.18)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ExplanationBundle",
    "FullCorpus",
    "compute_class1_shap",
    "explanation_candidate",
    "load_full_corpus",
    "load_full_rf_bundle",
    "train_full_rf",
    "write_explanation_package",
]
