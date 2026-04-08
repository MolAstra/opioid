#!/usr/bin/env python3

from opioid.explain import run_plip_rf_sar


if __name__ == "__main__":
    outputs = run_plip_rf_sar()
    print(f"saved: {outputs['interactions_csv']}")
    print(f"saved: {outputs['case_summary_csv']}")
    print(f"saved: {outputs['report_md']}")
    print(f"saved: {outputs['figures_dir']}")
    print(f"saved: {outputs['structures_dir']}")
    print(f"saved: {outputs['output_dir']}")
