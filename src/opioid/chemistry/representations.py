"""Deterministic tabular molecular representations for μORScreen.

The descriptor implementation intentionally follows Astra's KPGT feature
contract for ``RDKit2DNormalized``: descriptors are generated from canonical
isomeric SMILES, include exactly 200 values after the status field, and have
all non-finite values replaced with zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

MORGAN_RADIUS = 2
DESCRIPTOR_DIMENSION = 200
SMILES_COLUMN = "smiles"
LABEL_COLUMN = "label"


class FeatureRepresentationError(ValueError):
    """Raised when a requested molecular representation cannot be produced."""


class InvalidSmilesError(FeatureRepresentationError):
    """Raised when a SMILES value cannot be parsed into an RDKit molecule."""


class DescriptorGenerationError(FeatureRepresentationError):
    """Raised when DescriptaStorus does not satisfy the 200-value contract."""


@dataclass(frozen=True)
class FeatureSpec:
    """A versioned, explicit definition of one tabular molecular representation."""

    id: str
    description: str
    n_features: int
    morgan_bits: int | None = None
    include_normalized_descriptors: bool = False

    def __post_init__(self) -> None:
        expected_dimension = (self.morgan_bits or 0) + (
            DESCRIPTOR_DIMENSION if self.include_normalized_descriptors else 0
        )
        if not self.id:
            raise ValueError("Feature spec id must not be empty")
        if self.morgan_bits is not None and self.morgan_bits <= 0:
            raise ValueError("morgan_bits must be positive when it is specified")
        if expected_dimension == 0:
            raise ValueError("A feature spec must contain at least one representation component")
        if expected_dimension != self.n_features:
            raise ValueError(
                f"{self.id} has n_features={self.n_features}, expected "
                f"{expected_dimension} from its components"
            )

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Return the ordered components used to construct this feature vector."""

        components: list[str] = []
        if self.morgan_bits is not None:
            components.append(f"ecfp_{self.morgan_bits}")
        if self.include_normalized_descriptors:
            components.append("rdkit2d_normalized_200")
        return tuple(components)

    def manifest(self) -> dict[str, Any]:
        """Return serializable metadata suitable for model runtime manifests."""

        return {
            "id": self.id,
            "description": self.description,
            "n_features": self.n_features,
            "components": list(self.component_ids),
            "morgan_radius": MORGAN_RADIUS if self.morgan_bits is not None else None,
            "morgan_bits": self.morgan_bits,
            "descriptor": ("RDKit2DNormalized" if self.include_normalized_descriptors else None),
            "descriptor_dimension": (
                DESCRIPTOR_DIMENSION if self.include_normalized_descriptors else None
            ),
        }


_FEATURE_SPECS = {
    "ecfp_1024": FeatureSpec(
        id="ecfp_1024",
        description="Morgan/ECFP radius-2 binary fingerprint (1024 bits)",
        n_features=1024,
        morgan_bits=1024,
    ),
    "ecfp_2048": FeatureSpec(
        id="ecfp_2048",
        description="Morgan/ECFP radius-2 binary fingerprint (2048 bits)",
        n_features=2048,
        morgan_bits=2048,
    ),
    "rdkit2d_normalized_200": FeatureSpec(
        id="rdkit2d_normalized_200",
        description="DescriptaStorus RDKit2DNormalized descriptors (200 values)",
        n_features=DESCRIPTOR_DIMENSION,
        include_normalized_descriptors=True,
    ),
    "ecfp_1024_rdkit2d_normalized_200": FeatureSpec(
        id="ecfp_1024_rdkit2d_normalized_200",
        description="ECFP radius-2 1024 bits concatenated with 200 normalized descriptors",
        n_features=1024 + DESCRIPTOR_DIMENSION,
        morgan_bits=1024,
        include_normalized_descriptors=True,
    ),
    "ecfp_2048_rdkit2d_normalized_200": FeatureSpec(
        id="ecfp_2048_rdkit2d_normalized_200",
        description="ECFP radius-2 2048 bits concatenated with 200 normalized descriptors",
        n_features=2048 + DESCRIPTOR_DIMENSION,
        morgan_bits=2048,
        include_normalized_descriptors=True,
    ),
}

# Preserve registration order so benchmark result tables and plots are stable.
FEATURE_SPECS: Mapping[str, FeatureSpec] = MappingProxyType(_FEATURE_SPECS)


@dataclass(frozen=True)
class FeatureDataset:
    """Features and source-row bookkeeping after invalid SMILES are excluded."""

    features: np.ndarray
    labels: np.ndarray | None
    rows: pd.DataFrame
    invalid_smiles: list[str]
    feature_spec: FeatureSpec
    source_row_indices: list[Any]
    invalid_row_indices: list[Any]
    canonical_smiles: list[str]

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_invalid(self) -> int:
        return len(self.invalid_smiles)


def get_feature_spec(spec: FeatureSpec | str) -> FeatureSpec:
    """Resolve a registered feature spec from its object or stable identifier."""

    if isinstance(spec, FeatureSpec):
        return spec
    try:
        return FEATURE_SPECS[spec]
    except KeyError as error:
        expected = ", ".join(FEATURE_SPECS)
        raise ValueError(f"Unknown feature spec {spec!r}. Expected one of {expected}") from error


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a non-empty SMILES while retaining stereochemical information."""

    value = str(smiles).strip()
    if not value:
        raise InvalidSmilesError("SMILES must not be empty")
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise InvalidSmilesError(f"Invalid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def featurize_smiles(smiles: str, spec: FeatureSpec | str) -> np.ndarray:
    """Build one contiguous ``float32`` feature vector for a valid SMILES string."""

    feature_spec = get_feature_spec(spec)
    canonical = canonicalize_smiles(smiles)
    vector = _featurize_canonical_smiles(canonical, feature_spec)
    # Do not allow callers to mutate vectors cached for repeated canonical SMILES.
    return vector.copy()


def build_feature_dataset(
    df: pd.DataFrame,
    spec: FeatureSpec | str,
    *,
    smiles_column: str = SMILES_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> FeatureDataset:
    """Featurize valid rows while retaining labels, rows, and invalid-SMILES details.

    ``labels`` is ``None`` when ``label_column`` is not present, which supports
    inference data frames without adding a placeholder label column.
    """

    feature_spec = get_feature_spec(spec)
    if smiles_column not in df.columns:
        raise ValueError(f"Input dataframe is missing required SMILES column {smiles_column!r}")

    features: list[np.ndarray] = []
    kept_positions: list[int] = []
    kept_source_indices: list[Any] = []
    invalid_smiles: list[str] = []
    invalid_source_indices: list[Any] = []
    canonical_smiles: list[str] = []

    for position, (source_index, value) in enumerate(df[smiles_column].items()):
        raw_smiles = "" if pd.isna(value) else str(value)
        try:
            canonical = canonicalize_smiles(raw_smiles)
        except InvalidSmilesError:
            invalid_smiles.append(raw_smiles)
            invalid_source_indices.append(source_index)
            continue
        # Descriptor-contract failures are configuration/data errors, not
        # invalid SMILES. Let them fail the run rather than silently dropping
        # every affected row from a benchmark.
        vector = _featurize_canonical_smiles(canonical, feature_spec)

        kept_positions.append(position)
        kept_source_indices.append(source_index)
        canonical_smiles.append(canonical)
        features.append(vector)

    if features:
        feature_matrix = np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32)
    else:
        feature_matrix = np.zeros((0, feature_spec.n_features), dtype=np.float32)

    rows = df.iloc[kept_positions].reset_index(drop=True).copy()
    labels = _extract_labels(rows, label_column)
    return FeatureDataset(
        features=feature_matrix,
        labels=labels,
        rows=rows,
        invalid_smiles=invalid_smiles,
        feature_spec=feature_spec,
        source_row_indices=kept_source_indices,
        invalid_row_indices=invalid_source_indices,
        canonical_smiles=canonical_smiles,
    )


def _extract_labels(rows: pd.DataFrame, label_column: str) -> np.ndarray | None:
    if label_column not in rows.columns:
        return None
    try:
        numeric_labels = pd.to_numeric(rows[label_column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Label column {label_column!r} must contain numeric values") from error
    if numeric_labels.isna().any():
        raise ValueError(f"Label column {label_column!r} contains missing values")
    labels = numeric_labels.to_numpy(dtype=np.int64)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Label column {label_column!r} must contain only binary values 0 and 1")
    return labels


@lru_cache(maxsize=2)
def _morgan_generator(n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=n_bits,
    )


def _morgan_array(canonical_smiles: str, n_bits: int) -> np.ndarray:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:  # Defensive: canonicalize_smiles has already parsed it.
        raise InvalidSmilesError(f"Invalid canonical SMILES: {canonical_smiles!r}")
    fingerprint = _morgan_generator(n_bits).GetFingerprint(molecule)
    values = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


@lru_cache(maxsize=1)
def _descriptor_generator() -> Any:
    try:
        module = import_module("descriptastorus.descriptors.rdNormalizedDescriptors")
        return module.RDKit2DNormalized()
    except Exception as error:  # pragma: no cover - depends on optional installation.
        raise DescriptorGenerationError(
            "RDKit2DNormalized descriptors require the descriptastorus package"
        ) from error


def _normalized_descriptors(canonical_smiles: str) -> np.ndarray:
    """Generate Astra-compatible 200-dimensional RDKit2DNormalized descriptors."""

    values = _descriptor_generator().process(canonical_smiles)
    if values is None:
        raise DescriptorGenerationError("RDKit2DNormalized returned no descriptor values")
    try:
        n_values = len(values)
    except TypeError as error:
        raise DescriptorGenerationError(
            "RDKit2DNormalized returned a non-sequence result"
        ) from error
    if n_values != DESCRIPTOR_DIMENSION + 1:
        raise DescriptorGenerationError(
            "RDKit2DNormalized must return a status field plus exactly "
            f"{DESCRIPTOR_DIMENSION} descriptors; received {n_values} values"
        )
    if not bool(values[0]):
        raise DescriptorGenerationError("RDKit2DNormalized reported an unsuccessful status")
    try:
        descriptors = np.asarray(values[1:], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise DescriptorGenerationError("RDKit2DNormalized descriptors are not numeric") from error
    if descriptors.shape != (DESCRIPTOR_DIMENSION,):
        raise DescriptorGenerationError(
            "RDKit2DNormalized descriptor payload has unexpected shape "
            f"{descriptors.shape}; expected ({DESCRIPTOR_DIMENSION},)"
        )
    return np.ascontiguousarray(
        np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float32,
    )


@lru_cache(maxsize=20_000)
def _cached_feature_vector(canonical_smiles: str, spec_id: str) -> np.ndarray:
    feature_spec = get_feature_spec(spec_id)
    vector = _assemble_feature_vector(canonical_smiles, feature_spec)
    vector.setflags(write=False)
    return vector


def _featurize_canonical_smiles(canonical_smiles: str, spec: FeatureSpec) -> np.ndarray:
    registered_spec = FEATURE_SPECS.get(spec.id)
    if registered_spec is not spec and registered_spec != spec:
        # Custom specs are permitted only if their feature identity is not trying
        # to shadow a registered identifier. They bypass the global cache.
        if registered_spec is not None:
            raise FeatureRepresentationError(
                f"Custom feature spec conflicts with registered id {spec.id!r}"
            )
        return _assemble_feature_vector(canonical_smiles, spec)
    return _cached_feature_vector(canonical_smiles, spec.id)


def _assemble_feature_vector(canonical_smiles: str, spec: FeatureSpec) -> np.ndarray:
    components: list[np.ndarray] = []
    if spec.morgan_bits is not None:
        components.append(_morgan_array(canonical_smiles, spec.morgan_bits))
    if spec.include_normalized_descriptors:
        components.append(_normalized_descriptors(canonical_smiles))
    vector = components[0] if len(components) == 1 else np.concatenate(components)
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape != (spec.n_features,):
        raise FeatureRepresentationError(
            f"Feature spec {spec.id!r} produced shape {vector.shape}, expected ({spec.n_features},)"
        )
    return vector


__all__ = [
    "DESCRIPTOR_DIMENSION",
    "FEATURE_SPECS",
    "LABEL_COLUMN",
    "MORGAN_RADIUS",
    "SMILES_COLUMN",
    "DescriptorGenerationError",
    "FeatureDataset",
    "FeatureRepresentationError",
    "FeatureSpec",
    "InvalidSmilesError",
    "build_feature_dataset",
    "canonicalize_smiles",
    "featurize_smiles",
    "get_feature_spec",
]
