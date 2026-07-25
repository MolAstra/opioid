from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from opioid.benchmark.chemprop import (
    CHEMPROP_ACCELERATOR,
    CHEMPROP_DEVICES,
    ChempropRunner,
    chemprop_candidates,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "smiles": ["CC", "CCC", "CCO", "CCN"],
            "label": [0, 1, 0, 1],
        }
    )
    valid = pd.DataFrame(
        {
            "smiles": ["CCCC", "CCCO", "CCCN", "COC"],
            "label": [0, 1, 0, 1],
        }
    )
    return train, valid


def test_chemprop_cv_command_uses_explicit_train_validation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_0.pt").touch()

    monkeypatch.setattr("opioid.benchmark.chemprop.subprocess.run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    train, valid = _frames()
    spec = chemprop_candidates()[0]
    runner = ChempropRunner(executable="chemprop")
    checkpoints = runner.fit_fold(spec, train, valid, tmp_path / "chemprop")

    assert len(checkpoints) == 1
    command = commands[0]
    assert command.count("-i") == 1
    assert "--splits-column" in command
    assert "__split" in command
    assert command[command.index("--tracking-metric") + 1] == "val_loss"
    assert command[command.index("--accelerator") + 1] == CHEMPROP_ACCELERATOR
    assert command[command.index("--devices") + 1] == CHEMPROP_DEVICES
    assert runner.output_namespace == "gpu_devices_1_visible_7"
    input_path = Path(command[command.index("-i") + 1])
    input_frame = pd.read_csv(input_path)
    assert set(input_frame["__split"]) == {"train", "val"}
    assert "test" not in set(input_frame["__split"])
