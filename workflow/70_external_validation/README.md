# 70 — Exploratory external-case evaluation

Runs the fixed Workflow 60 LightGBM/TabPFN/RF ensemble on a workflow-owned literature-derived
source table containing 20 μOR ligands (10 antagonistic and 10 non-antagonistic).

Prerequisites:

```bash
./workflow/50_explain_rf/run.sh all --replace
./workflow/60_virtual_screen/run.sh all --replace
```

Run on physical GPU 5:

```bash
./workflow/70_external_validation/run.sh all --replace
```

The workflow audits stereochemistry-preserving canonical-SMILES overlap against all 982 labeled
molecules before inference. PZM21, cebranopadol, and endomorphin-1 are expected exact overlaps and
are retained in the 20-case table but excluded from the separate 17-case result.

The registered consensus remains strict: a molecule receives an antagonistic consensus label only
when all three uncalibrated class-1 scores are at least 0.5. The output reports every model and the
strict consensus using raw confusion counts, accuracy, sensitivity, specificity, and balanced
accuracy. AUROC and AUPRC are intentionally omitted because the source is a small, purposively
balanced literature case set. The compact HTML report provides the compound name, DOI, overlap
status, individual model predictions, and strict-consensus outcome for every case. Source SMILES
remain available in both prediction CSVs.

Owned state:

```text
data/external_mor_20.csv
results/
├── external_20_predictions.csv
├── external_17_nonoverlap_predictions.csv
├── overlap_audit.csv
├── accuracy_summary.csv
├── report.md
├── report.html
├── run_manifest.json
└── checksums.sha256
```

This workflow is exploratory rather than an independently sampled prospective validation. It
does not modify Workflow 60 models or tune their thresholds.

The complete non-model result package is published. Because model binaries are deliberately
excluded from Git, a fresh clone must regenerate Workflows 50 and 60 before rerunning this step.
