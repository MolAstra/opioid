"""DPBench local-partition integration for reviewer evaluation."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


TASK_ID = "muor_antagonism"
PROTOCOLS = ("scaffold", "random", "datasail")
SPLIT_SEED = 2026
N_FOLDS = 5
TEST_FRACTION = 0.10
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_DATA_ROOT = PROJECT_ROOT / "workflow" / "10_build_data"
DPBENCH_ROOT = BUILD_DATA_ROOT / "data" / "dpbench"
SOURCE_CSV = BUILD_DATA_ROOT / "data" / "source" / "opioid.csv"
DEFAULT_DPNET_COMMAND = "conda run --no-capture-output -n molm dpnet"

SMILES_COLUMN = "smiles"
LABEL_COLUMN = "label"
SOURCE_ROW_ID_COLUMN = "source_row_id"


@dataclass(frozen=True)
class DpbenchPaths:
    """Stable paths for one DPBench task and its local partitions."""

    root: Path
    task_id: str = TASK_ID

    @property
    def task_root(self) -> Path:
        return self.root / "task_pool" / self.task_id

    @property
    def raw_csv(self) -> Path:
        return self.task_root / "raw" / f"{self.task_id}.csv"

    @property
    def task_meta_yaml(self) -> Path:
        return self.task_root / "task_meta.yaml"

    def processed_dir(self, protocol: str) -> Path:
        _validate_protocols((protocol,))
        return self.task_root / f"processed_{protocol}"


def prepare_task(
    *,
    input_csv: Path | str = SOURCE_CSV,
    root: Path | str = DPBENCH_ROOT,
    task_id: str = TASK_ID,
    protocols: Iterable[str] = PROTOCOLS,
    replace: bool = False,
) -> DpbenchPaths:
    """Stage one DPBench task in the same root that holds its local partitions.

    A changed raw input cannot coexist with partitions built from an older
    snapshot.  Replacing such an input consequently requires a forced rebuild
    of all three reviewer protocols in the same invocation.
    """

    selected = tuple(protocols)
    _validate_protocols(selected)
    if not selected:
        raise ValueError("At least one DPBench protocol is required")

    source = Path(input_csv)
    frame = pd.read_csv(source)
    _validate_source_frame(frame, source)
    paths = DpbenchPaths(Path(root), task_id)
    raw_payload = _raw_payload(frame)
    meta_payload = _task_meta_yaml(task_id)
    changed = _task_inputs_changed(paths, raw_payload, meta_payload)
    existing_partitions = tuple(
        protocol for protocol in PROTOCOLS if paths.processed_dir(protocol).exists()
    )
    if changed and existing_partitions and (not replace or set(selected) != set(PROTOCOLS)):
        present = ", ".join(existing_partitions)
        raise ValueError(
            "DPBench task input changed while persisted partitions exist "
            f"({present}). Rebuild all protocols with --protocol all --replace."
        )

    paths.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_csv.write_text(raw_payload, encoding="utf-8")
    paths.task_meta_yaml.write_text(meta_payload, encoding="utf-8")
    return paths


def build_partitions(
    *,
    root: Path | str = DPBENCH_ROOT,
    task_id: str = TASK_ID,
    protocols: Iterable[str] = PROTOCOLS,
    dpnet_executable: str | Sequence[str] = DEFAULT_DPNET_COMMAND,
    replace: bool = False,
) -> DpbenchPaths:
    """Build one formal DPBench CV partition per reviewer protocol."""

    selected = tuple(protocols)
    _validate_protocols(selected)
    if not selected:
        raise ValueError("At least one DPBench protocol is required")
    paths = DpbenchPaths(Path(root), task_id)
    for protocol in selected:
        command = [
            *_command_prefix(dpnet_executable),
            "process",
            task_id,
            "--root",
            str(paths.root),
            "--method",
            protocol,
            "--layout",
            "cv",
            "--processed-dir",
            paths.processed_dir(protocol).name,
            "--cv-folds",
            str(N_FOLDS),
            "--test-fraction",
            f"{TEST_FRACTION:.2f}",
        ]
        if replace:
            command.append("--force")
        subprocess.run(command, check=True)
    return paths


def validate_partitions(
    *,
    root: Path | str = DPBENCH_ROOT,
    task_id: str = TASK_ID,
    protocols: Iterable[str] = PROTOCOLS,
    dpnet_executable: str | Sequence[str] = DEFAULT_DPNET_COMMAND,
) -> None:
    """Run DPBench's local-partition integrity validation for every protocol."""

    selected = tuple(protocols)
    _validate_protocols(selected)
    for protocol in selected:
        subprocess.run(
            [
                *_command_prefix(dpnet_executable),
                "validate",
                "--root",
                str(root),
                "--task",
                task_id,
                "--processed-dir",
                f"processed_{protocol}",
            ],
            check=True,
        )


def write_runtime_snapshot(
    *,
    root: Path | str = DPBENCH_ROOT,
    task_id: str = TASK_ID,
    dpnet_executable: str | Sequence[str] = DEFAULT_DPNET_COMMAND,
    datasail_source: Path | str | None = None,
) -> Path:
    """Persist split-tool versions and the exact staged task-input digests."""

    paths = DpbenchPaths(Path(root), task_id)
    destination = paths.root / "dpbench_runtime.json"
    datasail_path = Path(datasail_source).resolve() if datasail_source is not None else None
    snapshot = {
        "dpnet_version": _command_output([*_command_prefix(dpnet_executable), "--version"]),
        "datasail_source": str(datasail_path) if datasail_path is not None else None,
        "datasail_git_commit": (
            _command_output(["git", "-C", str(datasail_path), "rev-parse", "HEAD"])
            if datasail_path is not None
            else None
        ),
        "datasail_git_branch": (
            _command_output(["git", "-C", str(datasail_path), "branch", "--show-current"])
            if datasail_path is not None
            else None
        ),
        "task_id": task_id,
        "raw_csv_sha256": _sha256(paths.raw_csv),
        "task_meta_sha256": _sha256(paths.task_meta_yaml),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _raw_payload(frame: pd.DataFrame) -> str:
    payload = frame[[SMILES_COLUMN, LABEL_COLUMN]].copy()
    payload.insert(0, SOURCE_ROW_ID_COLUMN, range(len(payload)))
    return payload.to_csv(index=False)


def _task_inputs_changed(paths: DpbenchPaths, raw_payload: str, meta_payload: str) -> bool:
    if not paths.raw_csv.exists() or not paths.task_meta_yaml.exists():
        return True
    return (
        paths.raw_csv.read_text(encoding="utf-8") != raw_payload
        or paths.task_meta_yaml.read_text(encoding="utf-8") != meta_payload
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _command_prefix(executable: str | Sequence[str]) -> list[str]:
    if isinstance(executable, str):
        command = shlex.split(executable)
    else:
        command = list(executable)
    if not command:
        raise ValueError("DPBench executable command must not be empty")
    return command


def _validate_source_frame(frame: pd.DataFrame, source: Path) -> None:
    required = {SMILES_COLUMN, LABEL_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if frame.empty or frame[SMILES_COLUMN].isna().any():
        raise ValueError(f"{source} must contain non-empty SMILES values")
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="raise")
    if not labels.isin((0, 1)).all():
        raise ValueError(f"{source} must use binary labels 0 and 1")


def _validate_protocols(protocols: Iterable[str]) -> None:
    unknown = set(protocols).difference(PROTOCOLS)
    if unknown:
        expected = ", ".join(PROTOCOLS)
        raise ValueError(f"Unknown DPBench protocols: {sorted(unknown)}; expected {expected}")


def _task_meta_yaml(task_id: str) -> str:
    return f"""schema_version: 2
task_id: {task_id}
smiles_column: {SMILES_COLUMN}
id_column: {SOURCE_ROW_ID_COLUMN}
labels:
  - id: {LABEL_COLUMN}
    label_column: {LABEL_COLUMN}
    problem_type: binary
    value_type: int64
    num_classes: 2
    class_values: [0, 1]
    positive_class: 1
    unit: null
seed: {SPLIT_SEED}
split_method: random
split_config:
  fractions:
    train: 0.8
    valid: 0.1
    test: 0.1
extra_columns: []
"""
