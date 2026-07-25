from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from opioid.web import litserve_app


class _FakeRuntime:
    def predict_requests(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"input_smiles": row["smiles"], "status": "ok"} for row in requests]


def test_litapi_validation_batching_and_alignment() -> None:
    api = litserve_app.MuORConsensusAPI()
    api.runtime = _FakeRuntime()
    decoded = api.decode_request({"smiles": " CCO ", "explain": True})

    assert decoded == {"smiles": "CCO", "explain": True}
    batch = api.batch([decoded, {"smiles": "CCC", "explain": False}])
    assert api.unbatch(api.predict(batch)) == [
        {"input_smiles": "CCO", "status": "ok"},
        {"input_smiles": "CCC", "status": "ok"},
    ]
    assert api.max_batch_size == 32
    assert api.batch_timeout == pytest.approx(0.02)
    with pytest.raises(HTTPException) as missing:
        api.decode_request({"explain": False})
    assert missing.value.status_code == 422


def test_frontend_meta_and_single_only_routes(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    (frontend / "static").mkdir(parents=True)
    (frontend / "index.html").write_text("<h1>test frontend</h1>", encoding="utf-8")
    (frontend / "static" / "app.css").write_text("", encoding="utf-8")

    server = litserve_app.create_server(repo_root=tmp_path, frontend_root=frontend)
    paths = server.app.openapi()["paths"]
    predict_schema = paths["/api/v1/predict"]["post"]
    assert "requestBody" in predict_schema
    assert "parameters" not in predict_schema
    assert "/api/v1/predict/batch" not in paths
    assert "/api/v1/predict/csv" not in paths

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert "test frontend" in page.text
            meta = await client.get("/api/v1/meta")
            assert meta.json()["label_contract"]["1"] == "antagonistic"
            assert meta.json()["models"] == [
                "rf__ecfp_2048",
                "tabpfn__ecfp_2048",
                "lgbm__ecfp_1024_rdkit2d_normalized_200",
            ]
            assert meta.json()["limits"] == {"max_smiles_length": 4096}
            assert (await client.post("/api/v1/predict/batch", json={})).status_code == 404
            assert (await client.post("/api/v1/predict/csv")).status_code == 404

    asyncio.run(exercise())


def test_frontend_exposes_local_csv_batch_interface() -> None:
    frontend = Path("workflow/80_web_app/frontend")
    page = (frontend / "index.html").read_text(encoding="utf-8")
    script = (frontend / "static" / "app.js").read_text(encoding="utf-8")

    assert 'id="csv-input"' in page
    assert 'id="batch-predict-button"' in page
    assert 'id="batch-download"' in page
    assert 'data-panel="batch-panel"' in page
    assert "const MAX_BATCH_ROWS = 1000;" in script
    assert "const BATCH_CONCURRENCY = 32;" in script
    assert 'fetch("/api/v1/predict"' in script
    assert 'JSON.stringify({ smiles: value, explain: false })' in script
    assert "muor_all_three_antagonistic" in script
