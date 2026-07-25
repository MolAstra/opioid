# Screening source snapshots

These immutable snapshots are the exact Workflow 60 inputs. They are published so the cleaning
funnel and source-dependent screening results can be reproduced without machine-local paths.

| Snapshot | Rows | Main provenance |
| --- | ---: | --- |
| `gpcrdb.csv` | 12,090 | GPCRdb ligand export; Pándy-Szekeres et al., 2018, DOI `10.1093/nar/gkx1109` |
| `zinc_scaffold.csv` | 14,556 | Legacy focused ZINC snapshot; Sterling and Irwin, 2015, DOI `10.1021/acs.jcim.5b00559` |
| `reinvent_stage2_1.csv` | 90,000 | REINVENT generation output; Loeffler et al., 2024, DOI `10.1186/s13321-024-00812-5` |
| `opioid_ouroboros.csv` | 19,611 | OUROBOROS generation output associated with Wang et al., 2026, DOI `10.1002/advs.202513556` |

The retained ZINC filename is historical. The original construction log is unavailable, so this
repository does **not** describe the snapshot as scaffold-matched and cannot determine whether an
earlier collection step favored agonistic or antagonistic scaffolds. Results characterize only
this exact snapshot after the documented Workflow 60 cleaning process.

Run `sha256sum -c SHA256SUMS` from this directory to verify all four files. The repository MIT
license applies to μORScreen code, not automatically to third-party database snapshots; users
remain responsible for the upstream database and publication terms.
