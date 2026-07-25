"""Versioned reviewer-benchmark result manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opioid.benchmark.context import RunContext


RUNTIME_SCHEMA_VERSION = 2


def write_run_manifest(context: RunContext, payload: dict[str, Any]) -> Path:
    """Write reproducibility metadata for the isolated benchmark output."""

    path = context.output_dir / "run_manifest.json"
    full_payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "protocol": context.protocol,
        "split_dir": str(context.split_dir),
        "seed": context.seed,
        "threshold": context.threshold,
        **payload,
    }
    path.write_text(json.dumps(full_payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
