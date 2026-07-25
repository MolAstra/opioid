# μORScreen workflows

The numbered directories are the public, reproducible entry points for the reviewer workflow.
Run them in numeric order. Each workflow resolves paths from its own directory, so commands work
from any current working directory.

| Step | Purpose | Owned state |
| --- | --- | --- |
| `10_build_data` | Build and validate scaffold/random/DataSAIL 5CV + test partitions | `data/` |
| `20_audit_splits` | Run canonical DPNet split analysis plus fold diagnostics | `results/` |
| `30_benchmark_models` | Run the complete model × representation matrix | `results/` |
| `40_report_results` | Render comparison figures and the strict final report | `results/` |
| `50_explain_rf` | Refit and explain the RF/ECFP4-2048 screening model | `results/` |
| `60_virtual_screen` | Run the full-data RF/TabPFN/LightGBM consensus virtual screen | `data/`, `results/` |
| `70_external_validation` | Audit and score the fixed 20/17 literature case sets | `data/`, `results/` |
| `80_web_app` | Serve local LitServe/FastAPI predictions and RF explanations | frontend + API |

The Python package under `src/opioid/` contains shared scientific implementations. Workflow
directories contain orchestration, local defaults, data contracts, and operator documentation.
Data and non-model result evidence are version-controlled: CSV/JSON tables, row-level predictions,
figures, reports and manifests. Regenerable `.joblib`, `.ckpt`, `.pt`, logs and resume state are
ignored. Verify exact release coverage with:

```bash
python workflow/verify_release.py verify
```

Core RF explanation uses Workflow 50, candidate screening uses Workflow 60, exploratory external
case evaluation uses Workflow 70, and local web inference uses Workflow 80. A fresh clone must
regenerate the ignored Workflow 50/60 model binaries before running steps 70 or 80.
