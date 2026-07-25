# μORScreen v0.3.0 execution summary

This document records the current numbered Workflow evidence. Historical pre-Workflow outputs and
the previous long-form execution log are preserved locally under `tmp/archive/` and are not
inputs to the released results.

## Fixed data and split contract

- Source: 983 human μOR ligands; `label=1` means antagonistic and `label=0` means
  non-antagonistic.
- DPBench standardization removes one duplicate, leaving 982 molecules: 228 label 1 and 754
  label 0.
- Protocols: scaffold, random and DataSAIL; each has a fixed test and five train/validation folds.
- Split seed: 2026. Estimator seed: 42. Classification threshold: 0.5.
- Committed partition manifest SHA256 values:
  - scaffold: `a2de65184357a6b28962afb6e974006810852911ba8568ffbc245e038f71be34`
  - random: `e7813f050d4d44e87473480e41b948a2fcc6ed01ad0c40f6b9e663516068161e`
  - DataSAIL: `443b98148617a5e70b9293433f75f75d4908624c60fa3dab586f4147937dc8d7`

The partitions were generated with DPNet 0.13.0 and the DataSAIL revision recorded in
`workflow/10_build_data/data/dpbench/dpbench_runtime.json`. They were subsequently validated
read-only with DPNet 0.16.0.

## Benchmark and reporting

- Workflow 30 contains complete matrices for all three protocols.
- Each matrix contains 55 traditional model/representation candidates and three Chemprop
  candidates, with five validation and five shared-test evaluations per candidate.
- Metrics are AUROC, AUPRC, Accuracy, F1 and MCC with mean, sample SD and Student-t 95% CI.
- Workflow 40 reports all 58 candidates. Plot ordering is presentation only; there is no automatic
  Top-1 or deployment promotion.
- The fixed RF/ECFP4-2048 Workflow 50 explanation target was chosen using validation-only aggregate
  evidence and refit on all 982 labels. It has no new independent-test metric.

## Screening and external cases

- Workflow 60 first fixes one candidate in each of eight model families using DataSAIL validation
  AUROC, then uses shared-test AUROC to select LightGBM, TabPFN and RF for screening.
- This is explicitly a `test_informed_screening_ensemble`; the three uncalibrated scores are used
  for prioritization and strict all-three `score >= 0.5` consensus.
- The v0.3.0 GPU-5 rerun used an 8,192-molecule inference batch, standardized and globally
  deduplicated 95,868 non-training candidates. Eligible/source-level screening counts were 11,312
  GPCRdb, 7,041 ZINC, 58,056 REINVENT and 19,562 OUROBOROS molecules; strict consensus identified
  674, 9, 174 and 28 antagonistic candidates, respectively.
- Workflow 70 reports the fixed 20-case literature set and the 17-case cohort after excluding
  three exact training overlaps.
- The strict consensus has Accuracy/Balanced Accuracy of `0.700/0.700` on all 20 cases and
  `0.647/0.679` on the 17 non-overlapping cases.

## Publication and runtime boundary

- Git publishes all Workflow data plus non-model result evidence: tables, row-level predictions,
  figures, reports, manifests and checksums.
- `.joblib`, `.ckpt`, `.pt`, logs and resume state are local regenerable artifacts and are not
  published.
- Workflow 50 and 60 must be rerun before a fresh clone can execute Workflow 70 or start Workflow
  80.
- `python workflow/verify_release.py verify` validates exact coverage and SHA256 for the published
  data/result release.

## v0.3.0 validation record

The release was validated on 2026-07-25 against parent commit
`087ca5bc7caafe17280750e77421b33723f5bfb0`; the dirty state contained the intentional v0.3.0
release scope.

- Ruff: passed.
- Pytest: 71 passed, including all eight Workflow CLI help contracts.
- DPNet 0.16.0 read-only validation: scaffold, random and DataSAIL all valid with no warnings or
  errors.
- Workflow 30 strict validator: three complete 58-candidate matrices passed.
- Workflow 20 DPNet, Workflow 50, Workflow 60 and Workflow 70 package hashes: passed.
- Workflow 80: health, metadata and single-SMILES inference passed on GPU 5 at port 8002; browser
  CSV behavior is covered by pytest.
- `git diff --check`: passed. The staged-file size and publication-policy audits are performed
  immediately before commit.

The release inventory covers 1,488 files and 155,502,233 bytes. Exact hashes are stored in
`workflow/release_manifest.json` and `workflow/release_checksums.sha256`.
