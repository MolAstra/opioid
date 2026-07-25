# 30 — Benchmark models

Runs the frozen model × representation matrix against the three DPBench protocols. It reports all
candidates and never performs Top-1 selection or deployment promotion.

The formal complete run assigns scaffold/random/DataSAIL to physical GPUs 0/1/2 and runs the
three protocols concurrently:

```bash
./workflow/30_benchmark_models/run_full.sh
```

For diagnostics, one protocol may still be run directly with `run.sh`; it always writes beneath
this workflow's `results/` and cannot redirect benchmark artifacts to another output root.

Results are isolated at `results/<protocol>/benchmark/`; timestamped stdout/stderr logs are under
`results/_logs/`. Chemprop requires a visible GPU and uses one device. Resume state is tied to the
candidate definition, runtime, visible GPU namespace, and split snapshot. After all three
protocols complete, `validate.py` enforces 58 candidates, 290 validation rows, 290 shared-test rows,
finite five-metric summaries, fixed split hashes, and the no-Top-1 manifest contract.

Git publishes the result tables, per-fold predictions, Chemprop hyperparameters and manifests.
Chemprop `.ckpt`/`.pt` files, logs and `status.json` are local resume artifacts and are not
published.
