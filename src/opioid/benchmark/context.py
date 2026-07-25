"""Immutable paths and constants for a single benchmark protocol run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opioid.evaluation.dpbench import PROTOCOLS as DPBENCH_PROTOCOLS
from opioid.evaluation.dpbench import DPBENCH_ROOT, PROJECT_ROOT
from opioid.evaluation.dpbench import TASK_ID as DPBENCH_TASK_ID


SMILES_COLUMN = "smiles"
LABEL_COLUMN = "label"
THRESHOLD = 0.5
SEED = 42
N_FOLDS = 5
EVALUATION_PROTOCOLS = DPBENCH_PROTOCOLS
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workflow" / "30_benchmark_models" / "results"


@dataclass(frozen=True)
class RunContext:
    """All paths belonging to one isolated reviewer benchmark protocol."""

    protocol: str
    split_dir: Path
    output_dir: Path
    seed: int = SEED
    threshold: float = THRESHOLD

    @property
    def artifacts_dir(self) -> Path:
        return self.output_dir / "artifacts"

    @property
    def test_predictions_dir(self) -> Path:
        return self.output_dir / "test_predictions"

    @property
    def chemprop_dir(self) -> Path:
        return self.output_dir / "chemprop"

    @property
    def status_json(self) -> Path:
        return self.output_dir / "status.json"

    @classmethod
    def for_protocol(
        cls,
        protocol: str,
        *,
        split_root: Path | str = DPBENCH_ROOT,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        task_id: str = DPBENCH_TASK_ID,
        seed: int = SEED,
    ) -> "RunContext":
        if protocol not in EVALUATION_PROTOCOLS:
            expected = ", ".join(EVALUATION_PROTOCOLS)
            raise ValueError(f"Unknown protocol: {protocol}. Expected one of {expected}")
        split_root = Path(split_root)
        output_root = Path(output_root)
        return cls(
            protocol=protocol,
            split_dir=split_root / "task_pool" / task_id / f"processed_{protocol}",
            output_dir=output_root / protocol / "benchmark",
            seed=seed,
        )

    def ensure_output_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.artifacts_dir,
            self.test_predictions_dir,
            self.chemprop_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
