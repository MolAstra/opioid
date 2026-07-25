from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from opioid.benchmark.context import EVALUATION_PROTOCOLS, RunContext
from opioid.evaluation.dpbench import (
    DPBENCH_ROOT,
    N_FOLDS,
    PROTOCOLS,
    SOURCE_CSV,
    TASK_ID,
    build_partitions,
    prepare_task,
    validate_partitions,
    write_runtime_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "workflow" / "10_build_data" / "data" / "source" / "opioid.csv"


def test_default_data_paths_are_workflow_owned() -> None:
    workflow_root = REPO_ROOT / "workflow" / "10_build_data" / "data"
    assert SOURCE_CSV == workflow_root / "source" / "opioid.csv"
    assert DPBENCH_ROOT == workflow_root / "dpbench"


def test_committed_partitions_have_fixed_cv_contract() -> None:
    expected_test_rows = {"scaffold": 98, "random": 98, "datasail": 94}
    task_root = DPBENCH_ROOT / "task_pool" / TASK_ID
    for protocol in PROTOCOLS:
        split_dir = task_root / f"processed_{protocol}"
        assert (split_dir / "assignments.csv").is_file()
        assert (split_dir / "diagnostics.json").is_file()
        assert (split_dir / "manifest.json").is_file()
        test = pd.read_csv(split_dir / "test.csv")
        assert len(test) == expected_test_rows[protocol]
        assert set(test["label"]).issubset({0, 1})
        for fold in range(N_FOLDS):
            train = pd.read_csv(split_dir / f"train_fold-{fold}.csv")
            valid = pd.read_csv(split_dir / f"valid_fold-{fold}.csv")
            assert not train.empty and not valid.empty
            assert set(train["label"]).union(valid["label"]).issubset({0, 1})


def test_dpbench_task_input_preserves_raw_row_lineage(tmp_path: Path) -> None:
    paths = prepare_task(input_csv=DATASET_PATH, root=tmp_path / "dpbench")

    raw = pd.read_csv(paths.raw_csv)
    assert list(raw.columns) == ["source_row_id", "smiles", "label"]
    assert len(raw) == 983
    assert raw["source_row_id"].tolist() == list(range(983))
    meta = paths.task_meta_yaml.read_text(encoding="utf-8")
    assert "positive_class: 1" in meta
    assert "id_column: source_row_id" in meta


def test_dpbench_process_and_validation_commands_cover_all_three_protocols(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)

    monkeypatch.setattr("opioid.evaluation.dpbench.subprocess.run", fake_run)
    paths = build_partitions(root=tmp_path / "dpbench", dpnet_executable="dpnet-test", replace=True)
    validate_partitions(root=paths.root, dpnet_executable="dpnet-test")

    assert len(commands) == 6
    for protocol, command in zip(PROTOCOLS, commands[:3], strict=True):
        assert command == [
            "dpnet-test",
            "process",
            TASK_ID,
            "--root",
            str(paths.root),
            "--method",
            protocol,
            "--layout",
            "cv",
            "--processed-dir",
            f"processed_{protocol}",
            "--cv-folds",
            "5",
            "--test-fraction",
            "0.10",
            "--force",
        ]
    for protocol, command in zip(PROTOCOLS, commands[3:], strict=True):
        assert command == [
            "dpnet-test",
            "validate",
            "--root",
            str(paths.root),
            "--task",
            TASK_ID,
            "--processed-dir",
            f"processed_{protocol}",
        ]


def test_changed_dpbench_input_requires_all_protocols_and_replace(tmp_path: Path) -> None:
    root = tmp_path / "dpbench"
    paths = prepare_task(input_csv=DATASET_PATH, root=root)
    paths.raw_csv.write_text("source_row_id,smiles,label\n0,CC,0\n", encoding="utf-8")
    paths.processed_dir("random").mkdir(parents=True)

    with pytest.raises(ValueError, match="--protocol all --replace"):
        prepare_task(
            input_csv=DATASET_PATH,
            root=root,
            protocols=("random",),
            replace=True,
        )

    refreshed = prepare_task(input_csv=DATASET_PATH, root=root, replace=True)
    assert pd.read_csv(refreshed.raw_csv).shape == (983, 3)


def test_runtime_snapshot_has_no_implicit_datasail_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dpbench"
    prepare_task(input_csv=DATASET_PATH, root=root)
    monkeypatch.setattr(
        "opioid.evaluation.dpbench._command_output",
        lambda _command: "dpnet, version test",
    )

    snapshot_path = write_runtime_snapshot(root=root, dpnet_executable="dpnet-test")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot["datasail_source"] is None
    assert snapshot["datasail_git_commit"] is None
    assert snapshot["datasail_git_branch"] is None


def test_run_context_uses_dpbench_processed_layout(tmp_path: Path) -> None:
    context = RunContext.for_protocol("datasail", split_root=tmp_path, output_root=tmp_path / "out")
    assert EVALUATION_PROTOCOLS == PROTOCOLS
    assert context.split_dir == tmp_path / "task_pool" / TASK_ID / "processed_datasail"
    assert context.output_dir == tmp_path / "out" / "datasail" / "benchmark"
