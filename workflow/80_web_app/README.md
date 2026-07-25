# 80 — LitServe web application

Serves the Workflow 60 three-model consensus and on-demand Workflow 50 RF TreeSHAP explanation
through one local LitServe/FastAPI application.

Prerequisites:

```bash
./workflow/50_explain_rf/run.sh all --replace
./workflow/60_virtual_screen/run.sh all --replace
```

These steps are required after a fresh clone because model binaries are deliberately excluded
from Git even though their manifests and regeneration contracts are published.

Launch the API and frontend together on physical GPU 5:

```bash
./workflow/80_web_app/run_server.sh
```

Open `http://127.0.0.1:8000`. API documentation is available at `/docs`. Override local defaults
with `OPIOID_WEB_GPU`, `OPIOID_WEB_HOST`, `OPIOID_WEB_PORT`, or `OPIOID_CONDA_ENV`.

The inference endpoint is `POST /api/v1/predict` with a body such as:

```json
{"smiles": "CCO", "explain": true}
```

LitServe uses one GPU inference worker, batches up to 32 simultaneous molecule requests with a
20-ms collection window, and never duplicates the TabPFN model across Uvicorn workers. The JSON
interface intentionally accepts one molecule per request; concurrent requests are still combined
into GPU micro-batches internally. Invalid SMILES return `status=invalid_smiles`.

The browser also accepts a CSV containing a case-insensitive `smiles` column and at most 1,000
data rows (2 MB). It parses the file locally, submits up to 32 prediction requests concurrently
without TreeSHAP, previews the first 50 rows, and downloads a CSV that retains all original columns
and appends the three model scores/labels plus consensus fields. The uploaded file is not saved by
the server.

The displayed explanation is strictly the RF branch's local TreeSHAP decomposition. It is not an
explanation of TabPFN, LightGBM, or the final consensus. All three scores are uncalibrated, and the
ensemble remains `test_informed_screening_ensemble`. The main molecule uses a CoordGen SVG layout
with outline highlights; each reported Morgan bit includes SVG depictions of every mapped local
environment so fingerprint collisions remain visible.
