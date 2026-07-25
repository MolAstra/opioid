"""Shared molecular representations used by benchmark and inference code."""

from opioid.chemistry.representations import (
    DESCRIPTOR_DIMENSION,
    FEATURE_SPECS,
    MORGAN_RADIUS,
    FeatureDataset,
    FeatureRepresentationError,
    FeatureSpec,
    InvalidSmilesError,
    build_feature_dataset,
    canonicalize_smiles,
    featurize_smiles,
    get_feature_spec,
)

__all__ = [
    "DESCRIPTOR_DIMENSION",
    "FEATURE_SPECS",
    "MORGAN_RADIUS",
    "FeatureDataset",
    "FeatureRepresentationError",
    "FeatureSpec",
    "InvalidSmilesError",
    "build_feature_dataset",
    "canonicalize_smiles",
    "featurize_smiles",
    "get_feature_spec",
]
