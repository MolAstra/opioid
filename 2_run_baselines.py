#!/usr/bin/env python3

from opioid.baselines import run_all_baselines

BENCHMARK_MODELS = [
    "svm",
    "rf",
    "xgb",
    "lgbm",
    "tabpfn",
    "chemprop_default",
    "chemprop_morgan",
    "chemprop_morgan_rdkit2d",
]


def print_progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    print("Running baselines:", ", ".join(BENCHMARK_MODELS), flush=True)
    print("Ranking rule: cv_test > final_test > cv_valid", flush=True)
    results = run_all_baselines(
        benchmark_models=BENCHMARK_MODELS,
        progress=print_progress,
    )
    print(f"[done] cv summary: {results['cv_summary_csv']}", flush=True)
    print(f"[done] test metrics: {results['test_metrics_csv']}", flush=True)
    print(f"[done] runtime config: {results['runtime_config_json']}", flush=True)
