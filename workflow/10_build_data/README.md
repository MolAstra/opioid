# 10 — Build data

Owns the canonical μOR input table and the three fixed DPBench local partitions. The label
contract is immutable: `1 = antagonistic`, `0 = non-antagonistic`.

```bash
./workflow/10_build_data/run.sh --protocol all
./workflow/10_build_data/run.sh --protocol all --replace
```

The default `dpnet` command is `conda run --no-capture-output -n molm dpnet`. Pass
`--datasail-source /path/to/DataSAIL` when its Git checkout should be included in
`data/dpbench/dpbench_runtime.json`; no machine-local source path is assumed.

The committed partitions were generated with the tool versions recorded in that snapshot. A newer
installed `dpnet` may validate them, but must not silently rewrite them.

Inputs and outputs:

- source: `data/source/opioid.csv`
- fixed partitions: `data/dpbench/task_pool/muor_antagonism/processed_<protocol>/`
- runtime provenance: `data/dpbench/dpbench_runtime.json`
