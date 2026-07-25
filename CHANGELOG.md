# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.3.0] - 2026-07-25

### Added

- Eight ordered, self-contained subprojects under `workflow/`: data build, split audit, model
  benchmark, result reporting, RF explanation, virtual screening, external-case evaluation and
  local web inference.
- Version-controlled DPBench scaffold, fixed-seed random, and DataSAIL 5CV + fixed-test partitions
  owned by `workflow/10_build_data`, with their generating versions preserved in provenance.
- Canonical DPNet `analyze` split evidence with offline HTML, checksums, exact ECFP4/Tanimoto and
  DataSAIL leakage, supplemented by project-local per-fold train/validation diagnostics.
- A 55-candidate traditional ML × representation catalog plus three GPU Chemprop variants, with
  resumable CV/shared-test evaluation and complete five-metric comparison figures.
- A full-data RF/ECFP4-2048 TreeSHAP/SAR bundle with hashed provenance, Morgan collision evidence,
  representative local cases, and a reusable screening artifact.
- A workflow-owned four-library virtual screen using full-data RF/TabPFN/LightGBM models, canonical
  cleaning and overlap auditing, fixed all-three consensus, per-source predictions and checksums.
- A local LitServe/FastAPI application with automatic GPU micro-batching, a single-molecule JSON
  API, a browser-side CSV batch/download interface, and on-demand RF TreeSHAP explanations.
- A workflow-owned exploratory 20-case literature evaluation with exact training-overlap auditing,
  a separate 17-case non-overlap cohort, Accuracy/Balanced-Accuracy/confusion tables, and hashed
  provenance.
- Version-controlled source snapshots and non-model result evidence, plus a repository-wide
  SHA256 release inventory and pinned reviewer-environment constraints.

### Changed

- Replaced root scripts 1–3 with workflow-local `run.sh`/`main.py` entry points that resolve paths
  independently of the caller's current directory.
- Moved the canonical source table and persisted DPBench partitions out of the top-level
  `datasets/` directory and removed historical `datasets/cv`/`datasets/splits` layouts.
- Removed the duplicate legacy `opioid.baselines` training implementation; `opioid.benchmark` is
  now the only reviewer training stack.
- Removed the hard-coded local DataSAIL checkout path; source Git provenance is now optional and
  explicitly supplied.
- Separated plotting/report destinations from benchmark inputs so reporting cannot mutate training
  results.
- The reviewer matrix no longer performs or records Top-1 selection; every frozen candidate gets
  the same five-fold validation and shared-test treatment.
- Workflow 30 now owns and validates all benchmark artifacts, with a three-GPU complete-run entry
  point for scaffold/random/DataSAIL and no CLI path for redirecting training output elsewhere.
- Workflow 40 now gives all 58 candidates the same `model_name`/`model_type` presentation, writes
  a compact single-directory report package, and separates validation/test AUROC main figures.
- Workflow 40's optional family bars now fix one candidate per family by validation AUROC and
  reuse it across all displayed metrics instead of selecting separately from shared test.
- Replaced the legacy root RF/global-SHAP entry points with Workflow 50. Its 982-row refit is
  explicitly separated from independent benchmark metrics and emits an uncalibrated class-1
  screening score.
- Replaced the machine-local candidate-screening path with Workflow 60 source snapshots and an
  atomic result package. All eight families, including Chemprop, enter the disclosed two-stage
  DataSAIL selection; the resulting LightGBM/TabPFN/RF ensemble is explicitly test-informed and
  makes no new independent-test claim.
- Replaced Gradio web inference with Workflow 80. The service validates Workflow 50/60 artifacts,
  runs a single TabPFN GPU worker, and supports browser-side CSV prediction.
- Published all scientific tables, row-level predictions, figures, reports and manifests while
  keeping model binaries, Chemprop checkpoints, logs and resume state locally regenerable.

### Removed

- Removed the remaining root execution scripts, optional PDB/PLIP stack, legacy candidate
  screening/external-prediction modules and superseded runtime-promotion API. Numbered Workflows
  are now the only public entry points.

## [0.2.0] - 2026-07-20

### Added

- A repository handoff guide covering the reproducible `μORScreen` workflow, runtime contract, and operational prerequisites.

### Changed

- Updated the package and documented release state to version `0.2.0`.
- Ignore local `vendors/` checkouts so machine-local vendored sources do not appear in the release worktree.
