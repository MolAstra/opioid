"""Small reproducibility snapshots embedded in benchmark manifests."""

from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path

from opioid.benchmark.context import RunContext


_PACKAGES = ("numpy", "pandas", "scikit-learn", "rdkit", "descriptastorus", "chemprop")


def environment_snapshot() -> dict[str, object]:
    """Return the interpreter and relevant package versions without shelling out."""

    versions: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def split_snapshot(context: RunContext) -> dict[str, object]:
    """Record paths and hashes for persisted split source-of-truth artifacts."""

    paths = _shared_split_paths(context)
    paths.extend(_development_paths(context))
    paths.append(context.split_dir / "test.csv")
    return {
        "split_dir": str(context.split_dir),
        "files": {
            _snapshot_path_key(context, path): _sha256(path) if path.exists() else None
            for path in paths
        },
    }


def development_split_snapshot(context: RunContext) -> dict[str, object]:
    """Hash only train/validation inputs for a development-only artifact.

    The RF-SHAP anchor is deliberately trained without opening the independent
    test table.  Its provenance must preserve that distinction too, so this
    helper intentionally excludes ``test.csv`` and ``assignments.csv``.
    """

    paths = _development_paths(context)
    return {
        "split_dir": str(context.split_dir),
        "files": {
            str(path.relative_to(context.split_dir)): _sha256(path) if path.exists() else None
            for path in paths
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path_key(context: RunContext, path: Path) -> str:
    """Return a stable split-relative key, including sibling DPBench metadata."""

    try:
        return str(path.relative_to(context.split_dir))
    except ValueError:
        return f"../{path.relative_to(context.split_dir.parent)}"


def _development_paths(context: RunContext) -> list[Path]:
    """Return only train/validation files for the detected persisted layout."""

    if (context.split_dir / "manifest.json").exists():
        return [
            context.split_dir / f"{partition}_fold-{fold}.csv"
            for fold in range(5)
            for partition in ("train", "valid")
        ]
    return [
        context.split_dir / f"fold_{fold}" / name
        for fold in range(5)
        for name in ("train.csv", "valid.csv")
    ]


def _shared_split_paths(context: RunContext) -> list[Path]:
    if (context.split_dir / "manifest.json").exists():
        task_meta = context.split_dir.parent / "task_meta.yaml"
        return [
            context.split_dir / "manifest.json",
            context.split_dir / "assignments.csv",
            context.split_dir / "diagnostics.json",
            task_meta,
        ]
    return [context.split_dir / "assignments.csv", context.split_dir / "split_stats.json"]
