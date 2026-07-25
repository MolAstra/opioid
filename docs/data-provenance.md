# Data provenance and publication boundaries

## Labeled μOR task

The 983-row source table is the human μOR intrinsic-activity classification task from Oh et al.
(2024), DOI `10.1021/acschemneuro.4c00212`. The original annotations are retained:
`label=1` means antagonistic and `label=0` means non-antagonistic. DPBench standardization and
exact standardized-SMILES deduplication remove one duplicate, leaving 982 molecules.

The source table, raw DPBench task input, all three fixed 5CV + test partitions, manifests and
hashes are version-controlled under Workflow 10.

## Virtual-screening snapshots

Workflow 60 publishes exact GPCRdb, legacy ZINC, REINVENT and OUROBOROS input snapshots together
with SHA256 values. Their row-level outputs are also published. The legacy ZINC construction log
is unavailable; no scaffold-matching claim is made.

## Literature cases

Workflow 70 publishes the fixed 20-case source table with compound name, label and DOI. Exact
canonical-SMILES overlap against the 982 labeled molecules is disclosed in its result package.

## Licensing

The project MIT license covers μORScreen code. It does not relicense third-party publications or
database content. Users must independently follow the terms associated with Oh et al., GPCRdb,
ZINC, REINVENT and OUROBOROS.
