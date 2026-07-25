# Workflow 70 external-case evaluation

The fixed source contains 20 literature-derived μOR ligands (10 antagonistic and 10
non-antagonistic). Exact canonical-molecule auditing identified three training overlaps:
pzm21 (EXT-N05), cebranopadol (EXT-N08), endomorphin-1 (EXT-N10). The non-overlapping cohort therefore contains 17 cases (10 antagonistic and 7
non-antagonistic).

The strict LightGBM–TabPFN–RF consensus correctly classified **14/20**
(accuracy **0.700**; balanced accuracy **0.700**) before
overlap exclusion and **11/17** (accuracy **0.647**; balanced
accuracy **0.679**) after exclusion.

## Classification metrics and confusion counts

| cohort                 | predictor                    |   n |   n_antagonistic |   n_non_antagonistic |   tp |   fp |   tn |   fn |   correct |   accuracy |   sensitivity |   specificity |   balanced_accuracy |
|:-----------------------|:-----------------------------|----:|-----------------:|---------------------:|-----:|-----:|-----:|-----:|----------:|-----------:|--------------:|--------------:|--------------------:|
| external_20_all        | Random Forest                |  20 |               10 |                   10 |    9 |    2 |    8 |    1 |        17 |      0.850 |         0.900 |         0.800 |               0.850 |
| external_20_all        | TabPFN                       |  20 |               10 |                   10 |    8 |    1 |    9 |    2 |        17 |      0.850 |         0.800 |         0.900 |               0.850 |
| external_20_all        | LightGBM                     |  20 |               10 |                   10 |    7 |    1 |    9 |    3 |        16 |      0.800 |         0.700 |         0.900 |               0.800 |
| external_20_all        | Strict three-model consensus |  20 |               10 |                   10 |    5 |    1 |    9 |    5 |        14 |      0.700 |         0.500 |         0.900 |               0.700 |
| external_17_nonoverlap | Random Forest                |  17 |               10 |                    7 |    9 |    2 |    5 |    1 |        14 |      0.824 |         0.900 |         0.714 |               0.807 |
| external_17_nonoverlap | TabPFN                       |  17 |               10 |                    7 |    8 |    1 |    6 |    2 |        14 |      0.824 |         0.800 |         0.857 |               0.829 |
| external_17_nonoverlap | LightGBM                     |  17 |               10 |                    7 |    7 |    1 |    6 |    3 |        13 |      0.765 |         0.700 |         0.857 |               0.779 |
| external_17_nonoverlap | Strict three-model consensus |  17 |               10 |                    7 |    5 |    1 |    6 |    5 |        11 |      0.647 |         0.500 |         0.857 |               0.679 |

## Interpretation boundary

This is a small, purposively balanced literature case set rather than a random prospective sample.
The three overlapping non-antagonistic cases are disclosed, and the 17-case cohort is reported
separately. Accuracy and raw confusion counts are reported; AUROC and AUPRC are intentionally
omitted. Balanced accuracy is also reported as the unweighted mean of sensitivity and specificity.
The ensemble is the disclosed test-informed Workflow 60 screening artifact, and its scores are
uncalibrated.
