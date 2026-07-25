# μORScreen

`μORScreen` is a reproducible workflow for μ-opioid receptor ligand classification and reviewer
model comparison.

The label contract is fixed throughout the project:

- `label = 1`: antagonistic
- `label = 0`: non-antagonistic

## Environment

Python 3.12 is required. The main environment is shared by all numbered workflows.

```bash
conda create -n opioid_tmp python=3.12
conda activate opioid_tmp
pip install torch torchvision torchaudio
pip install -e .
```

DPBench/DataSAIL partitioning uses the separate `molm` environment. The default command is
`conda run --no-capture-output -n molm dpnet`. Chemprop reviewer jobs require a visible GPU.

## Repository layout

```text
opioid/
├── workflow/
│   ├── 10_build_data/          # canonical input and fixed DPBench partitions
│   ├── 20_audit_splits/        # similarity diagnostics
│   ├── 30_benchmark_models/    # 5CV + shared-test model matrix
│   ├── 40_report_results/      # model comparison figures and report
│   ├── 50_explain_rf/          # full-data RF TreeSHAP/SAR bundle
│   ├── 60_virtual_screen/      # full-data three-model virtual screen
│   ├── 70_external_validation/ # exploratory literature-case evaluation
│   └── 80_web_app/             # LitServe API and local research frontend
├── src/opioid/
│   ├── chemistry/              # molecular representation contracts
│   ├── evaluation/             # DPBench and split diagnostics
│   ├── benchmark/              # candidates, runners, metrics and reporting
│   ├── explain/
│   ├── screen/
│   └── web/
├── tests/
└── docs/
```

The numbered workflow directories are the public reviewer entry points. Each contains its own
`run.sh`, `main.py`, `README.md`, local defaults, and owned data/results. Shared scientific code
remains single-sourced under `src/opioid/`.

The numbered Workflow directories are the only public execution entry points. Pre-Workflow
scripts and outputs are retained only in the ignored local `tmp/archive/` area.

## Reviewer workflow

### 10 — Build and validate fixed data partitions

```bash
./workflow/10_build_data/run.sh --protocol all
```

Canonical state is owned by `workflow/10_build_data/data/`:

- `source/opioid.csv`: 983-row source table;
- `dpbench/task_pool/muor_antagonism/processed_<protocol>/`: fixed local partitions;
- `dpbench/dpbench_runtime.json`: tool and input provenance.

DPBench canonicalization/deduplication retains 982 molecules. Each of `scaffold`, `random`, and
`datasail` has one fixed test and five train/validation folds. Changed source input requires a
coherent rebuild with `--protocol all --replace`.

### 20 — Audit similarity and split isolation

```bash
./workflow/20_audit_splits/run.sh --protocol all
```

The canonical report under `results/<protocol>/dpnet_analysis/` comes directly from DPNet
`dpnet analyze`: self-contained HTML, JSON/CSV evidence, manifest and checksums covering data
quality, labels, molecular weight, scaffolds, exact development/test ECFP4/Tanimoto and
DataSAIL-compatible leakage. μORScreen separately writes the requested five train/validation fold
distributions under `fold_diagnostics/`. Scaffold isolation is asserted from the official report;
random and DataSAIL similarities are descriptive. An all-protocol run also writes the
title-free cross-protocol comparison to
`results/figures/test_to_development_max_tanimoto.png`.

### 30 — Run the complete model matrix

Run the formal complete matrix on physical GPUs 0–2:

```bash
./workflow/30_benchmark_models/run_full.sh
```

The traditional matrix contains 11 algorithms/presets × 5 representations = 55 candidates:

- SVM, RF, XGBoost, LightGBM, TabPFN;
- distance-weighted KNN (`k=3`, `k=5`);
- L1 and L2 logistic regression (`C=0.1`, `C=1`);
- ECFP-1024, ECFP-2048, RDKit2DNormalized-200, and both ECFP+descriptor combinations.

Three GPU Chemprop variants are evaluated alongside the traditional matrix. Every candidate gets
five validation scores and five scores on the protocol-specific shared test. AUROC, AUPRC,
Accuracy, F1 and MCC are summarized as mean, sample SD and Student-t 95% CI. The workflow never
selects Top-1 and never creates a deployment model.

Results are owned by `workflow/30_benchmark_models/results/<protocol>/benchmark/`. CV-only runs do
not read or hash `test.csv`; resume records are bound to candidate, runtime and split fingerprints.
The public CLI cannot redirect model artifacts outside Workflow 30. The full runner validates all
three complete result matrices before returning success.

### 40 — Plot and report all candidates

Workflow 40 reads only Workflow 30 results. All 58 candidates use the same visual treatment and
the same `model_name` + `model_type` identity; `candidate_id` is retained only for provenance.

```bash
./workflow/40_report_results/run.sh all --replace
```

The reporting workflow atomically replaces its own compact `results/` package without modifying
step 30. The two manuscript-facing figures are `figures/roc_valid.png` and
`figures/roc_test.png`; auxiliary PNGs cover all five metrics. The strict report requires all 58
candidates, all three protocols, five validation folds, and five shared-test evaluations per
candidate. It refuses incomplete/CV-only matrices and does not rank or select a model.

### 50 — Explain the full-data RF screening model

```bash
./workflow/50_explain_rf/run.sh all --replace
```

The user-approved explanation target is `Random Forest + ECFP4 (2048-bit)`. It has the highest
mean validation AUROC across the three frozen protocols (`0.8630`) but is not claimed to be
statistically superior. Workflow 50 refits it on all 982 deduplicated labels, verifies that all
three partition layouts reconstruct the same corpus, and writes TreeSHAP, Morgan-environment and
local SAR evidence.

Because the fixed test is included in this full-data refit, the resulting artifact has no new
independent performance metric. Workflow 30 remains the evaluation source of truth. The RF score
is uncalibrated and is intended for explanation and relative screening priority only.

### 60 — Run the consensus virtual screen

```bash
./workflow/60_virtual_screen/run.sh all --replace
```

Workflow 60 uses physical GPU 5 by default for TabPFN. It reuses Workflow 50's full-data
RF/ECFP4-2048 artifact and refits TabPFN/ECFP4-2048 and
LightGBM/ECFP4-1024+RDKit2D-200 on the same 982 labels. The four source snapshots are owned under
`workflow/60_virtual_screen/data/sources/`.
Invalid SMILES, canonical training overlap, within-source duplicates, and cross-source duplicate
membership are recorded before the fixed all-three `score >= 0.5` consensus is applied.

This ensemble is explicitly `test_informed_screening_ensemble`: DataSAIL validation AUROC first
fixes one candidate in each of all eight model families, including Chemprop, and shared-test AUROC
then selects LightGBM, TabPFN, and RF. Workflow 60 therefore makes no new independent test
performance claim; its uncalibrated probabilities rank candidates for experimental follow-up.

### 70 — Evaluate the exploratory literature cases

```bash
./workflow/70_external_validation/run.sh all --replace
```

Workflow 70 applies the unchanged Workflow 60 ensemble to a fixed 20-case literature source and
audits exact canonical overlap against all 982 labeled molecules. It reports the complete 20-case
table and a separate 17-case cohort after removing three disclosed non-antagonistic overlaps.
Accuracy, balanced accuracy and confusion counts are reported for all three models and the strict
consensus; AUROC and AUPRC are omitted. The compact HTML report includes names, DOI, overlap status
and molecule-level predictions; SMILES remain in the prediction CSVs. This small purposively
balanced case set is exploratory, not a representative prospective validation sample.

### 80 — Serve predictions and RF explanations

```bash
./workflow/80_web_app/run_server.sh
```

Workflow 80 starts one local LitServe/FastAPI application at `http://127.0.0.1:8000` and uses
physical GPU 5 by default. One GPU inference worker loads the Workflow 50/60 artifacts after
validating their hashes. LitServe merges simultaneous molecule requests into batches of up to 32.
The browser CSV interface accepts a `smiles` column with at most 1,000 molecules, submits up to 32
requests concurrently, and returns a downloadable table while keeping the uploaded file local.

The browser page and `/api/v1/predict` return all three uncalibrated scores and the fixed consensus.
Single-molecule requests can additionally return an additive RF TreeSHAP decomposition and a
contribution-highlighted structure. This explanation applies only to the RF branch, not to TabPFN,
LightGBM, or the consensus. API documentation is available at `/docs`.

## Reproducibility contract

- Split seed: `2026`; estimator seed: `42`; classification threshold: `0.5`.
- DataSAIL uses C1e/Morgan ECFP4-1024 allocation; no hard 0.70 Tanimoto claim is made.
- Source data, fixed partitions, tables, predictions, figures, reports and manifests are
  version-controlled.
- Regenerable `.joblib`, `.ckpt`, `.pt`, logs and resume state are ignored. A fresh clone must run
  Workflows 50 and 60 before Workflow 70 or 80.
- Verify the published evidence with `python workflow/verify_release.py verify`.
- Reviewer metrics must not be used for implicit deployment or automatic Top-1 selection.

See [workflow/README.md](workflow/README.md),
[docs/benchmark-runbook.md](docs/benchmark-runbook.md), and
[docs/reproducibility.md](docs/reproducibility.md) for operational details. Data sources and
publication boundaries are documented in [docs/data-provenance.md](docs/data-provenance.md);
the current execution record is [docs/execution-summary.md](docs/execution-summary.md).
