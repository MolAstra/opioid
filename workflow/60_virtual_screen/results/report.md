# Workflow 60 virtual-screen report

The workflow screened **95,868** globally unique, canonical, non-training molecules.
The fixed 0.5 all-three rule retained **870** putative antagonistic candidates
(0.91%). `label = 1` and every reported score mean antagonistic.

## Source summary

| source    |   n_source_rows |   n_invalid_smiles |   n_training_overlap_rows |   n_within_source_duplicate_rows |   n_unique_eligible |   n_feature_errors |   n_screened |   n_consensus_antagonistic |   consensus_antagonistic_rate |   n_consensus_non_antagonistic |   consensus_non_antagonistic_rate |
|:----------|----------------:|-------------------:|--------------------------:|---------------------------------:|--------------------:|-------------------:|-------------:|---------------------------:|------------------------------:|-------------------------------:|----------------------------------:|
| gpcrdb    |           12090 |                  0 |                       778 |                                0 |               11312 |                  0 |        11312 |                        674 |                        0.0596 |                           9339 |                            0.8256 |
| zinc      |           14556 |                  0 |                        24 |                             7491 |                7041 |                  0 |         7041 |                          9 |                        0.0013 |                           6371 |                            0.9048 |
| reinvent  |           90000 |                433 |                       743 |                            30768 |               58056 |                  0 |        58056 |                        174 |                        0.0030 |                          56882 |                            0.9798 |
| ouroboros |           19611 |                  1 |                        48 |                                0 |               19562 |                  0 |        19562 |                         28 |                        0.0014 |                          18466 |                            0.9440 |

## Ensemble

- Random Forest / ECFP4 (2048-bit), reused from Workflow 50.
- TabPFN / ECFP4 (2048-bit), refit on all 982 labels.
- LightGBM / ECFP4 (1024-bit) + RDKit2D (200-d), refit on all 982 labels.
- Consensus requires every uncalibrated class-1 score to be at least 0.5.
- The complete eight-family selection table is available in
  `datasail_two_stage_model_selection.html` and `.csv`.

## Interpretation boundary

This is a prioritization workflow, not an independent performance evaluation. DataSAIL validation
AUROC selected one candidate in each of eight model families, including Chemprop; shared-test
AUROC then selected the three families. This final ensemble is marked
`test_informed_screening_ensemble`, and its shared-test tables must not be presented as unused
independent validation. Scores are uncalibrated and require experimental follow-up.

Cleaning uses stereochemistry-preserving RDKit canonical SMILES for invalid-row detection,
within-source deduplication, cross-source prediction deduplication, and exclusion of all 982 labeled
training molecules. It does not neutralize charges or strip salts. Full row-level decisions are in
`cleaning_audit.csv`.
