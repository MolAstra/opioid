# 40 — Report results

Reads benchmark tables from `30_benchmark_models` without modifying them. The complete reporting
package is owned by this workflow under a single compact `results/` directory.

```bash
./workflow/40_report_results/run.sh all --replace
```

The formal package requires all three protocols, all 58 candidates, five validation folds, and
five shared-test evaluations per candidate. Generation happens in a staging directory and
`--replace` swaps it into place only after the package is complete.

All candidates have identical visual status. Tables expose separate `model_name` and `model_type`
columns and figures use those values as a two-line label; stable `candidate_id` values remain in
CSV outputs for provenance.

The output contract is:

```text
results/
├── all_candidates.csv
├── metrics_long.csv
├── validation_test_gap.csv
├── report.md
├── index.html
├── manifest.json
└── figures/                 # 21 PNG files
```

`figures/roc_valid.png` and `figures/roc_test.png` are the two primary AUROC figures. Each contains
all candidates across scaffold, random, and DataSAIL with values annotated to three decimals. The
remaining figures provide all-metric protocol overviews and validation-versus-test diagnostics.
The package contains no ranking or automatic model selection.

For an exploratory subset of figures, use `plot` with `--protocol` and `--stage`; `--metrics`
remains a compatibility alias for `--stage`. Formal `report` and `all` commands always require all
protocols and both stages.

## Optional validation-selected family bars

The standalone script below creates a separate descriptive view without changing the formal
58-candidate package or its existing 21 figures:

```bash
python workflow/40_report_results/plot_family_bars.py --replace
```

It writes four grouped bar charts (AUROC, AUPRC, Accuracy, and F1) plus a selection audit and
local manifest under `results/model_family_bars/`. The x-axis uses the compact family labels
SVM, RF, XGB, LightGBM, TabPFN, Chemprop, KNN, and LR. For each protocol and family, one complete
candidate—including the KNN or logistic-regression preset and molecular representation—is fixed
by the highest validation AUROC. Every metric then displays validation and shared-test values from
that same candidate. Figures use 600 DPI, Arial regular, two-decimal value
labels, and a three-column legend; the local manifest records these presentation parameters.

This is explicitly a `validation_selected_descriptive_family_view`. The selected candidate can
differ across protocols but not across displayed metrics, and the view is not a cross-family
Top-1 policy or deployment artifact. Use `selected_candidates.csv` to audit every displayed pair.
