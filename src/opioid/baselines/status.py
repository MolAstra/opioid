from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opioid.baselines import config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status() -> dict[str, Any]:
    if not config.STATUS_JSON.exists():
        return {"jobs": {}}
    return json.loads(config.STATUS_JSON.read_text(encoding="utf-8"))


def save_status(status: dict[str, Any]) -> None:
    config.STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.STATUS_JSON.write_text(json.dumps(status, indent=2), encoding="utf-8")


def get_job(status: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    return status.get("jobs", {}).get(job_id)


def is_completed(job: dict[str, Any] | None) -> bool:
    return bool(job and job.get("status") == "completed" and job.get("metrics"))


def mark_started(status: dict[str, Any], job_id: str, stage: str, model_label: str) -> None:
    status.setdefault("jobs", {})[job_id] = {
        "job_id": job_id,
        "stage": stage,
        "model_label": model_label,
        "status": "running",
        "updated_at": _utc_now(),
    }
    save_status(status)


def mark_failed(status: dict[str, Any], job_id: str, stage: str, model_label: str, error: str) -> None:
    status.setdefault("jobs", {})[job_id] = {
        "job_id": job_id,
        "stage": stage,
        "model_label": model_label,
        "status": "failed",
        "error": error,
        "updated_at": _utc_now(),
    }
    save_status(status)


def mark_completed(
    status: dict[str, Any],
    job_id: str,
    stage: str,
    model_label: str,
    metrics: dict[str, float],
    invalid_smiles: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "job_id": job_id,
        "stage": stage,
        "model_label": model_label,
        "status": "completed",
        "metrics": metrics,
        "invalid_smiles": invalid_smiles,
        "updated_at": _utc_now(),
    }
    if extra:
        payload.update(extra)
    status.setdefault("jobs", {})[job_id] = payload
    save_status(status)
