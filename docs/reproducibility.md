# Reproducing the published Workflow evidence

## Environments

The v0.3.0 evidence was produced on Linux with Python 3.12.13. Create the main environment and
install the CUDA 12.6 PyTorch build before installing this project:

```bash
conda create -n opioid_tmp -c conda-forge python=3.12.13 pip
conda activate opioid_tmp
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -c environment/reviewer-constraints.txt -e '.[dev]'
```

`environment/reviewer-constraints.txt` pins the scientific and web packages used for the
published evidence. Run manifests additionally record the relevant package versions, platform,
CUDA device and input hashes.

The committed partitions were generated with DPNet 0.13.0, RDKit 2025.09.3 and the DataSAIL
checkout recorded in `workflow/10_build_data/data/dpbench/dpbench_runtime.json`. Newer DPNet
versions may validate the fixed partitions but must not silently replace them.

## Published versus regenerable state

Git contains all source data, fixed partitions, result tables, row-level predictions, figures,
HTML/Markdown reports, manifests and checksums. It deliberately excludes model binaries,
Chemprop checkpoints, training logs and resume state.

Verify the published files without retraining:

```bash
python workflow/verify_release.py verify
conda run --no-capture-output -n molm dpnet validate \
  --root workflow/10_build_data/data/dpbench \
  --task muor_antagonism --processed-dir processed_scaffold
```

Repeat the DPNet command for `processed_random` and `processed_datasail`. Validate the complete
58-candidate matrices with:

```bash
conda run --no-capture-output -n opioid_tmp \
  python workflow/30_benchmark_models/validate.py
```

## Full regeneration

Run the numbered commands in `workflow/README.md`. Workflow 30 is the expensive GPU benchmark.
Workflow 50 and 60 regenerate the ignored RF, LightGBM and TabPFN model binaries; those local
artifacts are prerequisites for Workflow 70 and Workflow 80. Shared-test results must not be
reinterpreted as unused independent validation of the test-informed screening ensemble.
