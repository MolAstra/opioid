#!/usr/bin/env python3

from opioid.screen.plot_candidates import run_candidate_plots


if __name__ == "__main__":
    outputs = run_candidate_plots()
    print(f"saved: {outputs['summary_csv']}")
    print(f"saved: {outputs['hit_rate_png']}")
    print(f"saved: {outputs['count_png']}")
    print(f"saved: {outputs['ecdf_png']}")
    print(f"saved: {outputs['tail_png']}")
