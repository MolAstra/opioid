from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "workflow" / "verify_release.py"
MODULE_SPEC = importlib.util.spec_from_file_location("verify_release", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
verify_release = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(verify_release)


def _release_roots(tmp_path: Path) -> tuple[str, ...]:
    root = tmp_path / "workflow" / "10_build_data" / "data"
    root.mkdir(parents=True)
    (root / "source.csv").write_text("smiles,label\nCC,0\n", encoding="utf-8")
    (root / "model.joblib").write_bytes(b"local model")
    (root / "status.json").write_text("{}\n", encoding="utf-8")
    return ("workflow/10_build_data/data",)


def test_collect_publishable_files_excludes_runtime_artifacts(tmp_path: Path) -> None:
    roots = _release_roots(tmp_path)

    paths = verify_release.collect_publishable_files(tmp_path, roots)

    assert [path.name for path in paths] == ["source.csv"]


def test_release_inventory_detects_changed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _release_roots(tmp_path)
    monkeypatch.setattr(verify_release, "PUBLISHED_ROOTS", roots)
    payload = verify_release.write_release_inventory(tmp_path)

    assert payload["file_count"] == 1
    verify_release.verify_release_inventory(tmp_path)
    (tmp_path / "workflow/10_build_data/data/source.csv").write_text(
        "smiles,label\nCCC,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch"):
        verify_release.verify_release_inventory(tmp_path)
