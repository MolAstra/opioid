# 50 — Explain RF/ECFP4-2048

Builds one full-corpus random-forest bundle for core TreeSHAP/SAR interpretation and later
candidate-library screening.

```bash
./workflow/50_explain_rf/run.sh all --replace
```

The target is fixed to `Random Forest + ECFP4 (2048-bit)`. It has the highest mean validation
AUROC across scaffold, random, and DataSAIL in the frozen Workflow 30 matrix. This is a downstream,
user-approved explanation target; it does not change Workflow 30/40 into a Top-1 selection flow.

The model is refit on all 982 deduplicated labeled molecules. Consequently it has no independent
holdout metric. Workflow 30 remains the performance source of truth, while this model is used only
for explanation and later screening. Its class-1 score is not calibrated.

The workflow owns a compact `results/` package:

```text
results/
├── model/
│   ├── rf__ecfp_2048.joblib
│   └── model_manifest.json
├── figures/
│   ├── global_beeswarm.png
│   ├── global_importance.png
│   ├── class_prevalence.png
│   ├── bit_environments/
│   └── local_cases/
├── sample_index.csv
├── shap_values.npz
├── bit_summary.csv
├── bit_environments.csv
├── local_cases.csv
├── report.md
└── manifest.json
```

Local cases are training-corpus behavior examples, not validation/test errors. Morgan hash
collisions are retained as multiple environment rows rather than collapsed into one named motif.
Generation is staged and atomically swapped into place only after all checks succeed.

All explanation tables, SHAP arrays, figures, reports and manifests are published. The RF
`.joblib` is a regenerable local runtime artifact and is intentionally excluded from Git; run this
workflow before Workflows 60, 70 or 80 in a fresh clone.
