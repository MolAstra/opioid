# AGENTS.md

This handoff guide covers the current `μORScreen` reviewer workflow.

## Invariants

- Task: μ-opioid receptor ligand classification.
- Positive class: `label = 1` means `antagonistic`.
- Negative class: `label = 0` means `non-antagonistic`.
- Never reinterpret the label direction.
- Reviewer evaluation compares every registered candidate; it does not select Top-1 or silently
  create a deployment artifact.

## Canonical workflow

Run the numbered subprojects in order:

```bash
./workflow/10_build_data/run.sh --protocol all
./workflow/20_audit_splits/run.sh --protocol all
./workflow/30_benchmark_models/run.sh --protocol all --suite traditional --no-test
CUDA_VISIBLE_DEVICES=<gpu-id> ./workflow/30_benchmark_models/run.sh \
  --protocol all --suite chemprop --no-test
CUDA_VISIBLE_DEVICES=<gpu-id> ./workflow/30_benchmark_models/run.sh \
  --protocol all --suite traditional --suite chemprop
./workflow/40_report_results/run.sh all --replace
./workflow/50_explain_rf/run.sh all --replace
./workflow/60_virtual_screen/run.sh all --replace
CUDA_VISIBLE_DEVICES=<gpu-id> ./workflow/70_external_validation/run.sh all --replace
./workflow/80_web_app/run_server.sh
```

| Contract | Canonical location |
| --- | --- |
| Raw μOR table | `workflow/10_build_data/data/source/opioid.csv` |
| Fixed DPBench partitions | `workflow/10_build_data/data/dpbench/task_pool/muor_antagonism/processed_<protocol>/` |
| Canonical split evidence | `workflow/20_audit_splits/results/<protocol>/dpnet_analysis/` |
| Supplemental fold evidence | `workflow/20_audit_splits/results/<protocol>/fold_diagnostics/` |
| Reviewer model matrix | `workflow/30_benchmark_models/results/<protocol>/benchmark/` |
| Comparison figures/report | `workflow/40_report_results/results/` |
| Full-data RF explanation/screening bundle | `workflow/50_explain_rf/results/` |
| Screening source snapshots | `workflow/60_virtual_screen/data/sources/` |
| Three-model virtual-screen results | `workflow/60_virtual_screen/results/` |
| Exploratory external-case evaluation | `workflow/70_external_validation/results/` |
| Local LitServe/FastAPI application | `workflow/80_web_app/` |

Each workflow resolves defaults from its own file location. Do not reintroduce current-working-
directory-relative `datasets/` or `outputs/` paths into the reviewer core.

## Evaluation contract

- Protocols: `scaffold`, `random`, `datasail`.
- Each protocol has a fixed test plus five train/validation folds.
- Split seed: `2026`; estimator seed: `42`; threshold: `0.5`.
- Traditional candidates: 11 algorithm presets × five representations = 55.
- Chemprop candidates: graph, Morgan, and Morgan+RDKit2D; GPU only.
- Metrics: AUROC, AUPRC, Accuracy, F1 and MCC, with mean, sample SD and Student-t 95% CI.
- CV-only runs must not load or hash the independent test.
- A complete matrix evaluates every frozen candidate's five fold models on the shared test.
- Test metrics cannot drive reviewer tuning, thresholds, or claims of independent evaluation.
- The RF/ECFP4-2048 explanation target was explicitly chosen downstream from validation-only
  aggregate evidence; its full-data refit must not be assigned a new independent test metric.
- Workflow 60 uses DataSAIL validation AUROC to fix one candidate in each of all eight model
  families, including Chemprop, then shared-test AUROC to select LightGBM, TabPFN, and RF. The
  resulting `test_informed_screening_ensemble` is a disclosed screening-prioritization artifact;
  those test results are not unused independent validation of the ensemble.

Step 20 must use DPNet `dpnet analyze` as the source of truth for global split evidence. The local
similarity module is retained only for the extra five-fold train/validation view that fixed-test CV
analysis intentionally does not duplicate.

The strict report requires three complete 58-candidate matrices. Plot ordering by validation
metrics is presentation only, never model selection.

## Source layout

- `src/opioid/evaluation/`: DPBench integration and similarity diagnostics.
- `src/opioid/chemistry/`: ECFP and RDKit2DNormalized representation contracts.
- `src/opioid/benchmark/`: candidate catalog, estimators, runners, provenance and reporting.
- `src/opioid/explain/`: Workflow 50 full-corpus RF explanation.
- `src/opioid/screen/`: Workflow 60 screening and Workflow 70 external-case evaluation.
- `src/opioid/web/`: Workflow 80 artifact validation, consensus runtime, LitServe batching and
  FastAPI routes.

The duplicate legacy `src/opioid/baselines/` implementation has been removed. Do not recreate a
parallel training stack; extend `opioid.benchmark` instead.

## Environment

```bash
conda create -n opioid_tmp python=3.12
conda activate opioid_tmp
pip install torch torchvision torchaudio
pip install -e .
```

- DPBench/DataSAIL runs through `conda run --no-capture-output -n molm dpnet`.
- Pass `--datasail-source` only when a local DataSAIL checkout should be recorded in provenance;
  there is no hard-coded machine path.
- Chemprop, Workflow 60, Workflow 70, and Workflow 80 TabPFN require a GPU selected through
  `CUDA_VISIBLE_DEVICES`.

## Output and Git policy

- Workflow data and non-model results are version-controlled, including tables, predictions,
  figures, reports, manifests and checksums.
- `.joblib`, `.ckpt`, `.pt`, logs and resume state are ignored and must be regenerated locally.
- Historical scripts, outputs and experiments are retained only under ignored `tmp/archive/`.
- The numbered Workflow directories are the only public execution entry points.
- Before manuscript claims, record Git SHA, dirty state, commands, GPU/runtime versions, split
  manifests and run manifests in `docs/execution-summary.md`.

## Agent procedure

1. Read `README.md`, `workflow/README.md`, this file and the benchmark runbook.
2. Inspect the dirty worktree and preserve unrelated changes.
3. Verify all three committed DPBench partitions before changing evaluation behavior.
4. Treat each workflow's owned state as an explicit interface; do not write into an upstream step.
5. Run Ruff, pytest, CLI help checks, DPBench validation, release checksum verification and
   `git diff --check` before handoff.
6. Do not commit or push without explicit user approval.
