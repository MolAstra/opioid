# 60 — Virtual screen

Runs the full-data three-model virtual-screening ensemble against the four workflow-owned source
snapshots under `data/sources/`.

```bash
./workflow/60_virtual_screen/run.sh all --replace
```

The runner defaults to physical GPU 5 for TabPFN. Override it only when necessary:

```bash
OPIOID_SCREEN_GPU=4 ./workflow/60_virtual_screen/run.sh all --replace
```

The published v0.3.0 package used physical GPU 5 and a larger inference batch:

```bash
OPIOID_SCREEN_GPU=5 ./workflow/60_virtual_screen/run.sh all --replace --batch-size 8192
```

Batch size changes throughput and is recorded in `results/run_manifest.json`; it does not change
the frozen candidates, fitted training corpus, threshold, or consensus rule.

The ensemble is fixed to:

- Random Forest + ECFP4 (2048-bit), reused and checksum-validated from Workflow 50;
- TabPFN + ECFP4 (2048-bit), refit on all 982 labels;
- LightGBM + ECFP4 (1024-bit) + RDKit2D (200-d), refit on all 982 labels.

`label = 1` means antagonistic. A candidate is retained by the primary consensus rule only when all
three uncalibrated class-1 scores are at least `0.5`. Molecules are canonicalized with RDKit,
deduplicated within and across sources for prediction, and compared with all 982 labeled molecules
by canonical SMILES. Salt stripping and charge neutralization are deliberately not performed.

This is a `test_informed_screening_ensemble`. Within each of the eight benchmark model families,
including Chemprop, DataSAIL validation AUROC fixes one candidate. Shared-test AUROC then selects
the top three families. Therefore those test results are not unused independent validation of this
final ensemble, and Workflow 60 reports no new performance claim.

The workflow atomically owns `results/`:

```text
results/
├── models/
├── predictions/{gpcrdb,zinc,reinvent,ouroboros}.csv
├── figures/
├── all_screened_candidates.csv
├── consensus_antagonistic_candidates.csv
├── datasail_two_stage_model_selection.{csv,html}
├── cleaning_audit.csv
├── feature_errors.csv
├── screening_summary.csv
├── report.md
├── run_manifest.json
└── checksums.sha256
```

Source snapshots, row-level screening outputs, figures, reports and manifests are published.
The LightGBM/TabPFN `.joblib` files and the reused Workflow 50 RF artifact are not published.
`checksums.sha256` covers the public non-model evidence; model manifests retain artifact hashes
and the regeneration command.
