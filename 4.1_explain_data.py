#!/usr/bin/env python3

from opioid.explain import run_global_rf_shap


if __name__ == "__main__":
    outputs = run_global_rf_shap()
    print(f"saved: {outputs['predictions_csv']}")
    print(f"saved: {outputs['shap_values_csv']}")
    print(f"saved: {outputs['bit_summary_csv']}")
    print(f"saved: {outputs['top_examples_csv']}")
    print(f"saved: {outputs['report_md']}")
    print(f"saved: {outputs['beeswarm_png']}")
    print(f"saved: {outputs['importance_bar_png']}")
    print(f"saved: {outputs['prevalence_png']}")
    print(f"saved: {outputs['output_dir']}")
