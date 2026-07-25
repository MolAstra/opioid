"""LitServe/FastAPI application for the local μORScreen web workflow."""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import litserve as ls
from pydantic import BaseModel, Field

from opioid.web.consensus_runtime import (
    ConsensusRuntime,
    MAX_SMILES_LENGTH,
    WebArtifactPaths,
)


MAX_REQUEST_BYTES = 64 * 1024
LITSERVE_BATCH_SIZE = 32
LITSERVE_BATCH_TIMEOUT = 0.020
SERVICE_TIMEOUT_SECONDS = 300


class SingleRequest(BaseModel):
    """One-molecule body shown in OpenAPI and decoded by LitServe."""

    smiles: str = Field(min_length=1, max_length=MAX_SMILES_LENGTH)
    explain: bool = False


class MuORConsensusAPI(ls.LitAPI):
    """One-molecule LitServe API with automatic cross-request micro-batching."""

    def __init__(self, repo_root: Path | str | None = None) -> None:
        super().__init__(
            max_batch_size=LITSERVE_BATCH_SIZE,
            batch_timeout=LITSERVE_BATCH_TIMEOUT,
            api_path="/api/v1/predict",
        )
        self.repo_root = str(Path(repo_root).resolve()) if repo_root is not None else None
        self.runtime: ConsensusRuntime | None = None

    def setup(self, device: str) -> None:
        import torch

        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The Workflow 80 TabPFN service requires a visible CUDA GPU")
        self.runtime = ConsensusRuntime(WebArtifactPaths.from_repo(self.repo_root))

    def decode_request(self, request: SingleRequest) -> dict[str, Any]:
        if isinstance(request, SingleRequest):
            request = request.model_dump()
        if not isinstance(request, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object")
        if "smiles" not in request:
            raise HTTPException(status_code=422, detail="Request body must contain 'smiles'")
        smiles = request["smiles"]
        if not isinstance(smiles, str) or not smiles.strip():
            raise HTTPException(status_code=422, detail="SMILES must be a non-empty string")
        if len(smiles.strip()) > MAX_SMILES_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"SMILES must contain at most {MAX_SMILES_LENGTH} characters",
            )
        explain = request.get("explain", False)
        if not isinstance(explain, bool):
            raise HTTPException(status_code=422, detail="'explain' must be boolean")
        return {"smiles": smiles.strip(), "explain": explain}

    def batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return inputs

    def predict(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.runtime is None:
            raise RuntimeError("Consensus runtime has not been initialized")
        return self.runtime.predict_requests(requests)

    def unbatch(self, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return output


def create_server(
    *,
    repo_root: Path | str | None = None,
    frontend_root: Path | str | None = None,
) -> ls.LitServer:
    """Construct the single-GPU LitServer and mount local web routes."""

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).parents[3]
    frontend = (
        Path(frontend_root).resolve()
        if frontend_root is not None
        else root / "workflow" / "80_web_app" / "frontend"
    )
    index_path = frontend / "index.html"
    static_root = frontend / "static"
    if not index_path.is_file() or not static_root.is_dir():
        raise FileNotFoundError(f"Workflow 80 frontend assets are incomplete: {frontend}")

    api = MuORConsensusAPI(root)
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        timeout=SERVICE_TIMEOUT_SECONDS,
        max_payload_size=MAX_REQUEST_BYTES,
        model_metadata={
            "service": "muorscreen-consensus",
            "selection_scope": "test_informed_screening_ensemble",
            "label_1": "antagonistic",
            "scores_calibrated": False,
            "explanation_scope": "RF branch only",
        },
    )
    # LitServe registers a plain-text root route. Workflow 80 owns `/` as the
    # research UI while retaining LitServe's health/info/predict routes.
    server.app.router.routes = [
        route
        for route in server.app.router.routes
        if not (getattr(route, "path", None) == "/" and "GET" in getattr(route, "methods", set()))
    ]
    server.app.mount("/static", StaticFiles(directory=static_root), name="static")

    @server.app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(await asyncio.to_thread(index_path.read_text, encoding="utf-8"))

    @server.app.get("/api/v1/meta")
    async def metadata() -> dict[str, Any]:
        return {
            "service": "muorscreen-consensus",
            "label_contract": {"0": "non-antagonistic", "1": "antagonistic"},
            "models": [
                "rf__ecfp_2048",
                "tabpfn__ecfp_2048",
                "lgbm__ecfp_1024_rdkit2d_normalized_200",
            ],
            "threshold": 0.5,
            "consensus_rule": "all three scores >= 0.5",
            "selection_scope": "test_informed_screening_ensemble",
            "scores_calibrated": False,
            "explanation_scope": "RF branch only",
            "limits": {"max_smiles_length": MAX_SMILES_LENGTH},
        }

    return server


__all__ = [
    "MuORConsensusAPI",
    "SingleRequest",
    "create_server",
]
