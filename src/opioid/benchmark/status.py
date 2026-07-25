"""Small resumable status store keyed by protocol, candidate, and fold."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StatusStore:
    """Persist benchmark progress without coupling it to module globals."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 2, "jobs": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"schema_version": 2, "jobs": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")

    def completed(
        self,
        job_id: str,
        *,
        input_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a completed job only when its declared inputs still match."""

        job = self.payload.get("jobs", {}).get(job_id)
        if (
            isinstance(job, dict)
            and job.get("status") == "completed"
            and "metrics" in job
            and (input_fingerprint is None or job.get("input_fingerprint") == input_fingerprint)
        ):
            return job
        return None

    def start(self, job_id: str, **metadata: Any) -> None:
        self.payload.setdefault("jobs", {})[job_id] = {
            "job_id": job_id,
            "status": "running",
            "updated_at": _now(),
            **metadata,
        }
        self._save()

    def complete(self, job_id: str, metrics: dict[str, float], **metadata: Any) -> None:
        self.payload.setdefault("jobs", {})[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "metrics": metrics,
            "updated_at": _now(),
            **metadata,
        }
        self._save()

    def fail(self, job_id: str, error: str, **metadata: Any) -> None:
        self.payload.setdefault("jobs", {})[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "error": error,
            "updated_at": _now(),
            **metadata,
        }
        self._save()
