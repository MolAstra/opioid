# μOR functional-classification source table

`opioid.csv` contains 983 human μ-opioid receptor ligands from the functional-classification
task reported by:

> Oh M, Shen M, Liu R, Stavitskaya L, Shen J. Machine Learned Classification of Ligand
> Intrinsic Activities at Human μ-Opioid Receptor. *ACS Chemical Neuroscience* 2024,
> 15(15), 2842–2852. <https://doi.org/10.1021/acschemneuro.4c00212>

The project retains the published task labels. `label=1` means antagonistic and `label=0` means
non-antagonistic. DPBench normalization removes one duplicate and produces the committed
982-molecule partitions.

SHA256:

```text
5e2d53c299cf9d085f2f2f4cac42bb751eb1e0396926a0b978c10c5e1465666c  opioid.csv
```

The repository MIT license applies to μORScreen code, not automatically to third-party source
data. Users remain responsible for complying with the source publication and database terms.
