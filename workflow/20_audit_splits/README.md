# 20 — Audit splits

Reads the fixed partitions owned by `10_build_data`. The canonical evidence is produced by DPNet
0.14+ `dpnet analyze`; μORScreen adds only a supplemental five-fold train/validation view.

```bash
./workflow/20_audit_splits/run.sh --protocol all
./workflow/20_audit_splits/run.sh --protocol all --replace
./workflow/20_audit_splits/run.sh --protocol all --no-fold-diagnostics
```

For each protocol, `results/<protocol>/dpnet_analysis/` contains the official offline HTML,
machine-readable JSON/CSV tables, analysis manifest, and checksums. It covers data quality,
label/molecular-weight distributions, scaffold evidence, exact development/test ECFP4/Tanimoto,
DataSAIL-compatible leakage, and adapter diagnostics. `results/<protocol>/fold_diagnostics/`
contains the supplemental per-CV-fold train/validation distributions.

The scaffold hard assertion is read from the official DPNet report. Random and DataSAIL
similarities remain descriptive. Existing official reports are replaced atomically only with
`--replace`.

When all three protocols are audited together,
`results/figures/test_to_development_max_tanimoto.png` compares DPNet's held-out
test-to-development nearest-neighbor distributions. It preserves the original blue/gray/orange
protocol styling and dotted mean lines, without a figure title.
