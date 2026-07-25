#!/usr/bin/env python3
"""Build or verify the version-controlled μORScreen data/result release inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


WORKFLOW_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_ROOT.parent
CHECKSUM_PATH = Path("workflow/release_checksums.sha256")
MANIFEST_PATH = Path("workflow/release_manifest.json")
PUBLISHED_ROOTS = (
    "workflow/10_build_data/data",
    "workflow/20_audit_splits/results",
    "workflow/30_benchmark_models/results",
    "workflow/40_report_results/results",
    "workflow/50_explain_rf/results",
    "workflow/60_virtual_screen/data",
    "workflow/60_virtual_screen/results",
    "workflow/70_external_validation/data",
    "workflow/70_external_validation/results",
)
EXCLUDED_SUFFIXES = {".ckpt", ".joblib", ".log", ".pt", ".pyc"}
EXCLUDED_NAMES = {"status.json"}
EXCLUDED_PARTS = {"__pycache__", "_logs", "artifacts"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write or verify SHA256 coverage for all published Workflow data and non-model "
            "result evidence."
        )
    )
    parser.add_argument("command", choices=("write", "verify"))
    return parser.parse_args()


def collect_publishable_files(
    repo_root: Path = REPO_ROOT,
    roots: Iterable[str] | None = None,
) -> list[Path]:
    """Return the deterministic set covered by the public release checksum."""

    files: list[Path] = []
    for relative_root in PUBLISHED_ROOTS if roots is None else roots:
        root = repo_root / relative_root
        if not root.is_dir():
            raise FileNotFoundError(f"Missing published release root: {root}")
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and _is_publishable(path.relative_to(repo_root))
        )
    return sorted(set(files))


def write_release_inventory(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Write the public checksums and a compact publication-policy manifest."""

    paths = collect_publishable_files(repo_root)
    checksum_path = repo_root / CHECKSUM_PATH
    manifest_path = repo_root / MANIFEST_PATH
    lines = [f"{_sha256(path)}  {path.relative_to(repo_root)}" for path in paths]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "release": "0.3.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "published_data_and_non_model_result_evidence",
        "file_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "checksum_file": str(checksum_path.relative_to(repo_root)),
        "checksum_file_sha256": _sha256(checksum_path),
        "published_roots": list(PUBLISHED_ROOTS),
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "excluded_path_parts": sorted(EXCLUDED_PARTS),
        "model_artifacts_published": False,
        "regeneration": {
            "workflow_50": "./workflow/50_explain_rf/run.sh all --replace",
            "workflow_60": (
                "./workflow/60_virtual_screen/run.sh all --replace --batch-size 8192"
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_release_inventory(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Verify hashes, exact coverage, and the policy manifest."""

    checksum_path = repo_root / CHECKSUM_PATH
    manifest_path = repo_root / MANIFEST_PATH
    if not checksum_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Release inventory is missing; run `verify_release.py write`")
    expected = {
        path.relative_to(repo_root): path for path in collect_publishable_files(repo_root)
    }
    recorded: dict[Path, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"Malformed release checksum line: {line}")
        relative_path = Path(relative)
        if relative_path in recorded:
            raise ValueError(f"Duplicate release checksum path: {relative_path}")
        recorded[relative_path] = digest
    if set(recorded) != set(expected):
        missing = sorted(map(str, set(expected).difference(recorded)))
        extra = sorted(map(str, set(recorded).difference(expected)))
        raise ValueError(f"Release checksum coverage mismatch; missing={missing}, extra={extra}")
    mismatches = [
        str(relative)
        for relative, path in expected.items()
        if _sha256(path) != recorded[relative]
    ]
    if mismatches:
        raise ValueError(f"Release checksum mismatches: {mismatches}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("release") != "0.3.0"
        or manifest.get("file_count") != len(expected)
        or manifest.get("total_bytes")
        != sum(path.stat().st_size for path in expected.values())
        or manifest.get("checksum_file_sha256") != _sha256(checksum_path)
        or manifest.get("model_artifacts_published") is not False
    ):
        raise ValueError("Release manifest metadata does not match the published files")
    return manifest


def _is_publishable(relative_path: Path) -> bool:
    return (
        relative_path.suffix not in EXCLUDED_SUFFIXES
        and relative_path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(relative_path.parts)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    payload = (
        write_release_inventory()
        if args.command == "write"
        else verify_release_inventory()
    )
    print(
        f"release 0.3.0: {payload['file_count']} files, "
        f"{int(payload['total_bytes']) / 1024 / 1024:.1f} MiB"
    )


if __name__ == "__main__":
    main()
