"""Workflow 50/60 model loading, consensus inference and RF explanations."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D

from opioid.chemistry.representations import (
    FeatureRepresentationError,
    canonicalize_smiles,
    featurize_smiles,
)
from opioid.explain.full_rf import (
    _atoms_and_bonds,
    _contribution_color,
    _draw_highlighted_molecule,
    _environment_smiles,
    _morgan_example,
    _select_class1_expected_value,
    _select_class1_values,
)
from opioid.screen.virtual_screen import (
    LGBM_ID,
    RF_ID,
    SCORE_COLUMNS,
    TABPFN_ID,
    THRESHOLD,
    predict_lgbm_proba,
)


COMBINED_FEATURE_ID = "ecfp_2048_rdkit2d_normalized_200"
MAX_SMILES_LENGTH = 4096
DEFAULT_CACHE_SIZE = 10_000
DEFAULT_EXPLANATION_CACHE_SIZE = 256
TOP_EXPLANATION_BITS = 5


@dataclass(frozen=True)
class WebArtifactPaths:
    """Canonical model and explanation artifacts required by the web service."""

    repo_root: Path
    rf_bundle: Path
    screening_bundle: Path

    @classmethod
    def from_repo(cls, repo_root: Path | str | None = None) -> "WebArtifactPaths":
        root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        return cls(
            repo_root=root,
            rf_bundle=root / "workflow" / "50_explain_rf" / "results",
            screening_bundle=root / "workflow" / "60_virtual_screen" / "results",
        )


class ConsensusRuntime:
    """One-process runtime for the fixed Workflow 60 screening ensemble."""

    def __init__(
        self,
        paths: WebArtifactPaths | None = None,
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
        explanation_cache_size: int = DEFAULT_EXPLANATION_CACHE_SIZE,
    ) -> None:
        if cache_size <= 0 or explanation_cache_size <= 0:
            raise ValueError("Cache sizes must be positive")
        self.paths = paths or WebArtifactPaths.from_repo()
        self.cache_size = cache_size
        self.explanation_cache_size = explanation_cache_size
        self._prediction_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._explanation_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        manifests = self._validate_artifacts()
        self.rf_manifest = manifests["rf"]
        self.screening_manifest = manifests["screening"]
        self.model_manifest = manifests["models"]
        self.rf_model = joblib.load(self.paths.rf_bundle / self.rf_manifest["artifact_path"])
        model_root = self.paths.screening_bundle / "models"
        self.lgbm_model = joblib.load(
            model_root / self.model_manifest["models"][LGBM_ID]["artifact_path"]
        )
        self.tabpfn_model = joblib.load(
            model_root / self.model_manifest["models"][TABPFN_ID]["artifact_path"]
        )
        self.rf_explainer = shap.TreeExplainer(self.rf_model)
        self.rf_expected_value = _select_class1_expected_value(self.rf_explainer.expected_value)
        self.bit_summary = pd.read_csv(self.paths.rf_bundle / "bit_summary.csv").set_index("bit")

    def predict_requests(self, requests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict aligned request rows, vectorizing all uncached valid molecules."""

        normalized: list[dict[str, Any]] = []
        for request in requests:
            raw = str(request.get("smiles", "")).strip()
            explain = bool(request.get("explain", False))
            if not raw:
                raise ValueError("SMILES is required")
            if len(raw) > MAX_SMILES_LENGTH:
                raise ValueError(f"SMILES must contain at most {MAX_SMILES_LENGTH} characters")
            try:
                canonical = canonicalize_smiles(raw)
            except FeatureRepresentationError:
                normalized.append(
                    {
                        "input_smiles": raw,
                        "canonical_smiles": None,
                        "explain": explain,
                        "status": "invalid_smiles",
                    }
                )
                continue
            normalized.append(
                {
                    "input_smiles": raw,
                    "canonical_smiles": canonical,
                    "explain": explain,
                    "status": "ok",
                }
            )

        uncached = list(
            dict.fromkeys(
                row["canonical_smiles"]
                for row in normalized
                if row["status"] == "ok" and row["canonical_smiles"] not in self._prediction_cache
            )
        )
        if uncached:
            features = np.stack(
                [featurize_smiles(smiles, COMBINED_FEATURE_ID) for smiles in uncached], axis=0
            ).astype(np.float32, copy=False)
            rf_scores = self.rf_model.predict_proba(features[:, :2048])[:, 1]
            lgbm_scores = predict_lgbm_proba(self.lgbm_model, features)
            tabpfn_scores = self.tabpfn_model.predict_proba(features[:, :2048])[:, 1]
            for index, canonical in enumerate(uncached):
                self._cache_prediction(
                    canonical,
                    self._prediction_payload(
                        canonical=canonical,
                        rf_score=float(rf_scores[index]),
                        lgbm_score=float(lgbm_scores[index]),
                        tabpfn_score=float(tabpfn_scores[index]),
                    ),
                )

        results: list[dict[str, Any]] = []
        for row in normalized:
            if row["status"] != "ok":
                results.append(self._invalid_payload(str(row["input_smiles"])))
                continue
            canonical = str(row["canonical_smiles"])
            cached = dict(self._prediction_cache[canonical])
            self._prediction_cache.move_to_end(canonical)
            cached["input_smiles"] = row["input_smiles"]
            cached["explanation"] = self.explain(canonical) if row["explain"] else None
            results.append(cached)
        return results

    def explain(self, canonical_smiles: str) -> dict[str, Any]:
        """Return one additive RF TreeSHAP explanation and highlighted structure."""

        if canonical_smiles in self._explanation_cache:
            self._explanation_cache.move_to_end(canonical_smiles)
            return dict(self._explanation_cache[canonical_smiles])

        features = featurize_smiles(canonical_smiles, "ecfp_2048")[None, :]
        values = _select_class1_values(self.rf_explainer.shap_values(features))[0]
        rf_score = float(self.rf_model.predict_proba(features)[0, 1])
        reconstructed = float(self.rf_expected_value + values.sum())
        active = np.flatnonzero(features[0] > 0)
        positive = [
            int(bit)
            for bit in sorted(active, key=lambda bit: values[bit], reverse=True)
            if values[bit] > 0
        ][:TOP_EXPLANATION_BITS]
        negative = [
            int(bit) for bit in sorted(active, key=lambda bit: values[bit]) if values[bit] < 0
        ][:TOP_EXPLANATION_BITS]

        molecule, bit_info = _morgan_example(canonical_smiles)
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
        image = _draw_highlighted_molecule(
            molecule,
            atom_colors=atom_colors,
            bond_colors=bond_colors,
        )
        structure_svg = _draw_svg(
            molecule,
            width=760,
            height=460,
            atom_colors=atom_colors,
            bond_colors=bond_colors,
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = {
            "scope": "rf_branch_only",
            "base_value": self.rf_expected_value,
            "rf_score": rf_score,
            "reconstructed_rf_score": reconstructed,
            "additivity_residual": abs(rf_score - reconstructed),
            "positive_contributions": self._contribution_rows(molecule, bit_info, values, positive),
            "negative_contributions": self._contribution_rows(molecule, bit_info, values, negative),
            "structure_png_data_url": (
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            ),
            "structure_svg_data_url": _svg_data_url(structure_svg),
            "color_legend": {
                "green": "positive contribution toward antagonistic RF score",
                "red": "negative contribution away from antagonistic RF score",
                "orange": "overlapping positive and negative environments",
            },
        }
        self._explanation_cache[canonical_smiles] = payload
        self._explanation_cache.move_to_end(canonical_smiles)
        while len(self._explanation_cache) > self.explanation_cache_size:
            self._explanation_cache.popitem(last=False)
        return dict(payload)

    def info(self) -> dict[str, Any]:
        return {
            "service": "muorscreen-consensus",
            "label_contract": {"0": "non-antagonistic", "1": "antagonistic"},
            "threshold": THRESHOLD,
            "consensus_rule": "all three model scores >= 0.5",
            "selection_scope": "test_informed_screening_ensemble",
            "scores_calibrated": False,
            "explanation_scope": "RF branch only",
            "models": [RF_ID, TABPFN_ID, LGBM_ID],
        }

    def _contribution_rows(
        self,
        molecule: Chem.Mol,
        bit_info: dict[int, list[tuple[int, int]]],
        values: np.ndarray,
        bits: list[int],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bit in bits:
            environments = sorted(
                {
                    _environment_smiles(molecule, int(center), int(radius))
                    for center, radius in bit_info.get(bit, [])
                }
            )
            environment_svgs = [
                _svg_data_url(_draw_fragment_svg(environment)) for environment in environments
            ]
            summary = self.bit_summary.loc[bit] if bit in self.bit_summary.index else None
            rows.append(
                {
                    "bit": bit,
                    "shap_value": float(values[bit]),
                    "environment_smiles": environments,
                    "environment_svg_data_urls": environment_svgs,
                    "global_mean_abs_shap": (
                        float(summary["mean_abs_shap"]) if summary is not None else None
                    ),
                    "global_carrier_count": (
                        int(summary["carrier_count"]) if summary is not None else None
                    ),
                    "antagonistic_prevalence": (
                        float(summary["antagonistic_prevalence"]) if summary is not None else None
                    ),
                    "non_antagonistic_prevalence": (
                        float(summary["non_antagonistic_prevalence"])
                        if summary is not None
                        else None
                    ),
                }
            )
        return rows
    def _validate_artifacts(self) -> dict[str, dict[str, Any]]:
        rf_manifest_path = self.paths.rf_bundle / "model" / "model_manifest.json"
        screening_manifest_path = self.paths.screening_bundle / "run_manifest.json"
        model_manifest_path = self.paths.screening_bundle / "models" / "model_manifest.json"
        required = [
            rf_manifest_path,
            screening_manifest_path,
            model_manifest_path,
            self.paths.rf_bundle / "bit_summary.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing web model prerequisites. Run Workflow 50 and Workflow 60 first: "
                + ", ".join(missing)
            )
        rf = json.loads(rf_manifest_path.read_text(encoding="utf-8"))
        screening = json.loads(screening_manifest_path.read_text(encoding="utf-8"))
        models = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if rf.get("candidate", {}).get("candidate_id") != RF_ID:
            raise ValueError("Workflow 50 manifest is not the approved RF artifact")
        if screening.get("selection_scope") != "test_informed_screening_ensemble":
            raise ValueError("Workflow 60 manifest has an unexpected selection scope")
        if set(models.get("models", {})) != {RF_ID, LGBM_ID, TABPFN_ID}:
            raise ValueError("Workflow 60 model manifest does not contain the fixed ensemble")

        rf_path = self.paths.rf_bundle / str(rf["artifact_path"])
        if _sha256(rf_path) != rf.get("artifact_sha256"):
            raise ValueError("Workflow 50 RF artifact checksum mismatch")
        model_root = self.paths.screening_bundle / "models"
        for candidate_id in (LGBM_ID, TABPFN_ID):
            entry = models["models"][candidate_id]
            artifact = model_root / str(entry["artifact_path"])
            if _sha256(artifact) != entry.get("artifact_sha256"):
                raise ValueError(f"Workflow 60 artifact checksum mismatch: {candidate_id}")
        return {"rf": rf, "screening": screening, "models": models}

    def _cache_prediction(self, canonical: str, payload: dict[str, Any]) -> None:
        self._prediction_cache[canonical] = payload
        self._prediction_cache.move_to_end(canonical)
        while len(self._prediction_cache) > self.cache_size:
            self._prediction_cache.popitem(last=False)

    @staticmethod
    def _prediction_payload(
        *,
        canonical: str,
        rf_score: float,
        lgbm_score: float,
        tabpfn_score: float,
    ) -> dict[str, Any]:
        scores = {
            SCORE_COLUMNS[RF_ID]: rf_score,
            SCORE_COLUMNS[TABPFN_ID]: tabpfn_score,
            SCORE_COLUMNS[LGBM_ID]: lgbm_score,
        }
        labels = {
            "rf_pred_label": int(rf_score >= THRESHOLD),
            "tabpfn_pred_label": int(tabpfn_score >= THRESHOLD),
            "lightgbm_pred_label": int(lgbm_score >= THRESHOLD),
        }
        values = list(scores.values())
        votes = sum(labels.values())
        return {
            "input_smiles": canonical,
            "canonical_smiles": canonical,
            "status": "ok",
            **scores,
            **labels,
            "consensus_all_antagonistic": int(votes == 3),
            "consensus_all_non_antagonistic": int(votes == 0),
            "consensus_vote_count": votes,
            "consensus_mean_score": float(np.mean(values)),
            "consensus_min_score": float(np.min(values)),
            "consensus_max_score": float(np.max(values)),
            "model_score_range": float(np.max(values) - np.min(values)),
            "explanation": None,
        }

    @staticmethod
    def _invalid_payload(input_smiles: str) -> dict[str, Any]:
        empty = {
            SCORE_COLUMNS[RF_ID]: None,
            SCORE_COLUMNS[TABPFN_ID]: None,
            SCORE_COLUMNS[LGBM_ID]: None,
            "rf_pred_label": None,
            "tabpfn_pred_label": None,
            "lightgbm_pred_label": None,
            "consensus_all_antagonistic": None,
            "consensus_all_non_antagonistic": None,
            "consensus_vote_count": None,
            "consensus_mean_score": None,
            "consensus_min_score": None,
            "consensus_max_score": None,
            "model_score_range": None,
        }
        return {
            "input_smiles": input_smiles,
            "canonical_smiles": None,
            "status": "invalid_smiles",
            **empty,
            "explanation": None,
        }


def _prepared_2d_molecule(molecule: Chem.Mol) -> Chem.Mol:
    """Return a copy with a clean CoordGen layout suitable for browser rendering."""

    prepared = Chem.Mol(molecule)
    prepared.RemoveAllConformers()
    rdCoordGen.AddCoords(prepared)
    return rdMolDraw2D.PrepareMolForDrawing(prepared)


def _draw_svg(
    molecule: Chem.Mol,
    *,
    width: int,
    height: int,
    atom_colors: dict[int, tuple[float, float, float]] | None = None,
    bond_colors: dict[int, tuple[float, float, float]] | None = None,
) -> str:
    """Draw a crisp, lightly highlighted SVG without obscuring atom labels."""

    atom_colors = atom_colors or {}
    bond_colors = bond_colors or {}
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.padding = 0.12
    options.bondLineWidth = 1.8
    options.fillHighlights = False
    options.continuousHighlight = False
    options.highlightRadius = 0.22
    options.highlightBondWidthMultiplier = 5
    drawer.DrawMolecule(
        _prepared_2d_molecule(molecule),
        highlightAtoms=sorted(atom_colors),
        highlightBonds=sorted(bond_colors),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _draw_fragment_svg(smiles: str) -> str:
    """Draw a local Morgan environment, including non-sanitizable aromatic fragments."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        # Morgan environments are open graph fragments rather than complete
        # molecules. Aromatic fragments such as ``c(c)c`` are valid SMARTS but
        # intentionally fail standalone SMILES sanitization.
        molecule = Chem.MolFromSmarts(smiles)
    if molecule is None:
        return ""
    return _draw_svg(molecule, width=220, height=130)


def _svg_data_url(svg: str) -> str | None:
    if not svg:
        return None
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ConsensusRuntime", "MAX_SMILES_LENGTH", "WebArtifactPaths"]
