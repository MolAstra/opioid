#!/usr/bin/env python3

from opioid.explain.rf_sar import run_rf_sar


if __name__ == "__main__":
    outputs = run_rf_sar()
    print(f"saved: {outputs['sar_index_csv']}")
    print(f"saved: {outputs['sar_summary_csv']}")
    print(f"saved: {outputs['sar_report_md']}")
    print(f"saved: {outputs['output_dir']}")
