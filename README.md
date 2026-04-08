# μORScreen

Machine-learning workflow for `μ`-opioid receptor ligand classification, SAR interpretation, virtual screening, and lightweight deployment.

The current project is organized around a reproducible sequence of top-level scripts:

- scaffold-aware data splitting
- baseline model benchmarking
- publication-style plotting
- RF-SHAP interpretation
- structure-assisted SAR comparison
- multi-database candidate screening
- web prediction
- small external validation on curated literature ligands

Important label convention:

- `label = 1`: antagonistic
- `label = 0`: non-antagonistic

## Environment

Python `3.12` is required.

Typical setup:

```bash
conda create -n opioid_tmp python=3.12
conda activate opioid_tmp

# Install a PyTorch build that matches your machine.
pip install torch torchvision torchaudio

pip install -e .
```

Optional environment for PLIP analysis:

```bash
conda create -n plip_env python=3.12
conda activate plip_env
pip install plip
```

`4.3_plip_rf_sar.py` currently expects the PLIP executable to be available in `plip_env`.

## Repository Layout

```text
opioid/
├── datasets/
│   ├── opioid.csv
│   └── cv/
├── src/opioid/
│   ├── baselines/
│   ├── explain/
│   ├── screen/
│   └── web/
├── 1_build_data.py
├── 2_run_baselines.py
├── 3_plot_baselines.py
├── 4_explain_rf.py
├── 4.1_explain_data.py
├── 4.2_compare_pdb_rfshap.py
├── 4.3_plip_rf_sar.py
├── 5_build_candidate_library.py
├── 5.1_plot_candidates.py
├── 6_webserver.py
├── 7_novel_mols.py
└── outputs/
```

### Directory Guide

`datasets/`

- `opioid.csv`: master dataset used to build splits.
- `cv/`: generated scaffold-based train/valid/test files and split statistics.

`src/opioid/baselines/`

- model training, inference, feature generation, metrics, artifact writing.

`src/opioid/explain/`

- RF local SAR, global SHAP, PDB alignment, and PLIP comparison workflows.

`src/opioid/screen/`

- candidate-library prediction, candidate plotting, and external-validation utilities.

`src/opioid/web/`

- RF web inference backend used by the Gradio app.

`outputs/baselines/`

- benchmark metrics, saved models, runtime config, per-model test predictions, and baseline figures.

`outputs/explain_rf/`

- local RF-SHAP SAR examples, molecule-level figures, summary tables, and report.

`outputs/explain_data/`

- global RF-SHAP tables, top-bit examples, and manuscript-ready global figures.

`outputs/explain_pdb/`

- literature-backed PDB case comparison against RF-SHAP, including ligand PNGs.

`outputs/explain_plip_rf/`

- PLIP interaction summaries and RF-SAR figures for selected structural cases.

`outputs/candidates/`

- per-database screening predictions, candidate statistics, and screening figures.

`outputs/novel_mols/`

- formal external validation set and prediction outputs for curated literature ligands.

`outputs/tmp_novel_mols/`

- automatic backup copy of `outputs/novel_mols/`.

## End-to-End Workflow

### 1. Build scaffold-based splits

```bash
python 1_build_data.py
```

Outputs:

- `datasets/cv/test.csv`
- `datasets/cv/fold_*/train.csv`
- `datasets/cv/fold_*/valid.csv`
- `datasets/cv/split_stats.json`

### 2. Run baseline models

```bash
python 2_run_baselines.py
```

Current benchmark set:

- `svm`
- `rf`
- `xgb`
- `lgbm`
- `tabpfn`
- `chemprop_default`
- `chemprop_morgan`
- `chemprop_morgan_rdkit2d`

Key outputs:

- `outputs/baselines/cv5_summary_metrics.csv`
- `outputs/baselines/test_metrics.csv`
- `outputs/baselines/artifacts/runtime_config.json`

`runtime_config.json` is the deployment contract used by downstream scripts, including candidate screening, web inference, and external validation.

### 3. Plot baseline benchmark figures

```bash
python 3_plot_baselines.py
```

Current outputs are grouped bar plots for:

- AUROC
- AUPRC
- Accuracy
- F1

They are written to `outputs/baselines/figures/`.

### 4. RF interpretation workflows

Local RF-SHAP SAR:

```bash
python 4_explain_rf.py
```

Global RF-SHAP analysis:

```bash
python 4.1_explain_data.py
```

PDB-to-RF-SHAP comparison:

```bash
python 4.2_compare_pdb_rfshap.py
```

PLIP + RF-SAR comparison:

```bash
python 4.3_plip_rf_sar.py
```

Notes:

- `4_explain_rf.py` focuses on selected success and failure molecules.
- `4.1_explain_data.py` summarizes global RF bit importance over the full dataset.
- `4.2_compare_pdb_rfshap.py` compares representative `μOR` ligands from PDB against RF-SHAP interpretation.
- `4.3_plip_rf_sar.py` combines receptor interaction summaries with ligand-side RF-SHAP SAR.

### 5. Candidate screening

Build per-database ranked candidate libraries:

```bash
python 5_build_candidate_library.py
```

Plot screening summary figures:

```bash
python 5.1_plot_candidates.py
```

Current screening sources are defined in `src/opioid/screen/candidates.py` under `DB_SOURCES`.

Important behavior:

- screening uses the current `top3` models from `outputs/baselines/artifacts/runtime_config.json`
- training, validation, and test overlaps are removed before screening outputs are written
- results are kept as ranked full-library outputs, not truncated hit lists

Main outputs:

- `outputs/candidates/pred_gpcrdb.csv`
- `outputs/candidates/pred_zinc.csv`
- `outputs/candidates/pred_reinvent.csv`
- `outputs/candidates/pred_ouroboros.csv`
- `outputs/candidates/candidate_stats.csv`
- `outputs/candidates/figures/`

### 6. RF webserver

```bash
python 6_webserver.py
```

Features:

- single SMILES prediction
- batch CSV prediction
- CSV must contain `smiles` or `SMILES`
- CSV row limit: `1000`
- output fields:
  - `antagonist_probability`
  - `pred_label`
  - `status` for batch mode

This app uses the saved best model from `outputs/baselines/artifacts/runtime_config.json`.

### 7. External validation on curated literature molecules

```bash
python 7_novel_mols.py
```

This workflow:

- reads `outputs/novel_mols/external_mor_20_raw.csv`
- canonicalizes SMILES
- removes overlaps with `train/valid/test`
- predicts with the current top-3 ensemble
- writes formal outputs to `outputs/novel_mols/`
- mirrors them to `outputs/tmp_novel_mols/` as backup

Main outputs:

- `outputs/novel_mols/external_mor_20.csv`
- `outputs/novel_mols/external_top3_predictions.csv`
- `outputs/novel_mols/external_top3_summary.csv`
- `outputs/novel_mols/external_top3_analysis.md`

## Current Public Interfaces

Top-level scripts are the main public entrypoints. Internally:

- `src/opioid/baselines/` provides training and artifact generation utilities.
- `src/opioid/screen/candidates.py` exposes the top-model predictor and candidate-library workflow.
- `src/opioid/web/rf_web.py` exposes `RFWebPredictor`.
- `src/opioid/explain/` provides reusable explanation workflows behind scripts `4*`.

## Reproducibility Notes

- Classification threshold is currently `0.5`.
- Morgan fingerprint settings are currently radius `2` and `2048` bits.
- The saved runtime config is the source of truth for deployed prediction workflows.
- Candidate screening and external validation are both designed to avoid leakage from `train/valid/test`.

## Practical Notes

- The candidate screening source paths are currently local absolute paths; adjust `DB_SOURCES` before running on another machine.
- PLIP raw outputs are available even if PyMOL-based rendered PLIP figures are unavailable in the current environment.
- The repository contains generated `outputs/` artifacts from the active analysis workflow; they are part of the working project state, not just temporary cache.
