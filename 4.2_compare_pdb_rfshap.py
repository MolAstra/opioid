#!/usr/bin/env python3

from opioid.explain import run_pdb_rfshap_alignment


if __name__ == "__main__":
    outputs = run_pdb_rfshap_alignment()
    print(f"saved: {outputs['alignment_csv']}")
    print(f"saved: {outputs['alignment_md']}")
    print(f"saved: {outputs['output_dir']}")
