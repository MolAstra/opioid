from __future__ import annotations

from pathlib import Path

from opioid.benchmark.context import RunContext
from opioid.benchmark.provenance import split_snapshot


def test_split_snapshot_includes_sibling_dpbench_task_metadata(tmp_path: Path) -> None:
    task_root = tmp_path / "task_pool" / "muor_antagonism"
    split_dir = task_root / "processed_scaffold"
    split_dir.mkdir(parents=True)
    for name in ("manifest.json", "assignments.csv", "diagnostics.json", "test.csv"):
        (split_dir / name).write_text(name, encoding="utf-8")
    for fold in range(5):
        for partition in ("train", "valid"):
            (split_dir / f"{partition}_fold-{fold}.csv").write_text(
                f"{partition}-{fold}", encoding="utf-8"
            )
    (task_root / "task_meta.yaml").write_text("task: muor_antagonism\n", encoding="utf-8")

    snapshot = split_snapshot(
        RunContext(protocol="scaffold", split_dir=split_dir, output_dir=tmp_path / "output")
    )

    assert snapshot["files"]["../task_meta.yaml"] is not None
    assert snapshot["files"]["manifest.json"] is not None
