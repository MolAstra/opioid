#!/usr/bin/env python3

from opioid.screen.novel_mols import run_novel_molecule_prediction


if __name__ == "__main__":
    outputs = run_novel_molecule_prediction()
    print(f"saved: {outputs['clean_csv']}")
    print(f"saved: {outputs['predictions_csv']}")
    print(f"saved: {outputs['summary_csv']}")
    print(f"saved: {outputs['analysis_md']}")
    print(f"saved: {outputs['overlap_report_csv']}")
    print(f"saved: {outputs['invalid_smiles_json']}")
