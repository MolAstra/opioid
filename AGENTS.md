# AGENTS.md

This file is a handoff guide for another agent or clean environment to reproduce the `μORScreen` workflow.

## Core Semantics

- Task: `μ`-opioid receptor ligand classification
- Positive class: `label = 1` means `antagonistic`
- Negative class: `label = 0` means `non-antagonistic`
- Primary deployment contract: `outputs/baselines/artifacts/runtime_config.json`

Do not reinterpret the label direction. Several downstream analyses depend on `1 = antagonistic`.

## Project Entry Points

Top-level scripts are the public workflow:

```bash
python 1_build_data.py
python 2_run_baselines.py
python 3_plot_baselines.py
python 4_explain_rf.py
python 4.1_explain_data.py
python 4.2_compare_pdb_rfshap.py
python 4.3_plip_rf_sar.py
python 5_build_candidate_library.py
python 5.1_plot_candidates.py
python 6_webserver.py
python 7_novel_mols.py
```

Use them in this order unless you only need a downstream artifact.

## Environment

Primary environment:

```bash
conda create -n opioid_tmp python=3.12
conda activate opioid_tmp
pip install torch torchvision torchaudio
pip install -e .
```

PLIP environment:

```bash
conda create -n plip_env python=3.12
conda activate plip_env
pip install plip
```

Notes:
- `4.3_plip_rf_sar.py` calls `conda run -n plip_env plip ...`
- `6_webserver.py` depends on `gradio`
- `4_explain_rf.py` and `4.1_explain_data.py` depend on `shap`

## Minimal Reproduction Paths

### A. Full benchmark reproduction

Use this when no trained artifacts exist yet.

```bash
python 1_build_data.py
python 2_run_baselines.py
python 3_plot_baselines.py
```

Expected key outputs:
- `datasets/cv/test.csv`
- `datasets/cv/fold_*/train.csv`
- `datasets/cv/fold_*/valid.csv`
- `outputs/baselines/cv5_summary_metrics.csv`
- `outputs/baselines/test_metrics.csv`
- `outputs/baselines/artifacts/runtime_config.json`

### B. Interpretation reproduction

Requires existing baseline artifacts, especially:
- `outputs/baselines/artifacts/rf.joblib`
- `outputs/baselines/artifacts/runtime_config.json`

Run:

```bash
python 4_explain_rf.py
python 4.1_explain_data.py
python 4.2_compare_pdb_rfshap.py
python 4.3_plip_rf_sar.py
```

Key output roots:
- `outputs/explain_rf`
- `outputs/explain_data`
- `outputs/explain_pdb`
- `outputs/explain_plip_rf`

### C. Candidate screening reproduction

Requires trained baseline artifacts and the screening source databases referenced in `src/opioid/screen/candidates.py`.

Run:

```bash
python 5_build_candidate_library.py
python 5.1_plot_candidates.py
```

Outputs:
- `outputs/candidates/pred_*.csv`
- `outputs/candidates/candidate_stats.csv`
- `outputs/candidates/candidate_plot_summary.csv`
- `outputs/candidates/figures/*.png`

### D. Web inference

Requires existing baseline artifacts.

Run:

```bash
python 6_webserver.py
```

Behavior:
- single SMILES or CSV input
- CSV must contain `smiles` or `SMILES`
- maximum `1000` rows
- output columns include `antagonist_probability` and `pred_label`

### E. External validation

Requires existing baseline artifacts.

Run:

```bash
python 7_novel_mols.py
```

Input:
- `outputs/novel_mols/external_mor_20_raw.csv`

Formal outputs:
- `outputs/novel_mols/external_mor_20.csv`
- `outputs/novel_mols/external_top3_predictions.csv`
- `outputs/novel_mols/external_top3_summary.csv`
- `outputs/novel_mols/external_top3_analysis.md`

Backup mirror:
- `outputs/tmp_novel_mols/*`

## Important Path Assumptions

Some paths are machine-local and must be changed on a new system.

Most important:
- `src/opioid/screen/candidates.py`
  - `DB_SOURCES["gpcrdb"]["path"]`
  - `DB_SOURCES["zinc"]["path"]`
  - `DB_SOURCES["reinvent"]["path"]`
  - `DB_SOURCES["ouroboros"]["path"]`

If these source files do not exist on the new machine, `5_build_candidate_library.py` will fail immediately.

## Source of Truth Files

- Model/runtime contract: `outputs/baselines/artifacts/runtime_config.json`
- Baseline summary: `outputs/baselines/cv5_summary_metrics.csv`
- External validation raw set: `outputs/novel_mols/external_mor_20_raw.csv`
- Candidate DB definitions: `src/opioid/screen/candidates.py`
- Web limits and CSV rules: `src/opioid/web/rf_web.py`

Prefer reading these files before changing workflow behavior.

## Git and Release State

- Package version: `0.1.0`
- Git tag: `v0.1.0`
- Repository: `git@github.com:MolAstra/opioid.git`
- License: MIT

Generated artifacts under `outputs/`, local caches, and `dist/` are ignored by git.

## Operational Warnings

- `outputs/` is not version-controlled. If a clean environment needs downstream scripts, regenerate `outputs/baselines` first.
- `4.3_plip_rf_sar.py` may produce PLIP raw reports without rendered PLIP pictures if PyMOL is unavailable in `plip_env`.
- `7_novel_mols.py` copies formal outputs into `outputs/tmp_novel_mols` automatically; `outputs/novel_mols` remains the formal result directory.
- Candidate screening removes overlap with `train/valid/test` before writing final candidate files.

## Recommended Agent Workflow

If the goal is to audit or continue the project:

1. Read `README.md`
2. Read this `AGENTS.md`
3. Check whether `outputs/baselines/artifacts/runtime_config.json` exists
4. If it does not exist, start from `1_build_data.py` and `2_run_baselines.py`
5. If it exists, use it as the deployment source of truth and only rerun the needed downstream scripts

This repository is organized so that most downstream work can be reproduced without retraining, as long as `outputs/baselines/artifacts/` is present.
