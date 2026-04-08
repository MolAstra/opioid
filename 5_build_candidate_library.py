#!/usr/bin/env python3

from opioid.screen.candidates import DB_SOURCES, run_candidate_library


if __name__ == "__main__":
    print("Building per-database candidate libraries from:")
    for db_name, source_path in DB_SOURCES.items():
        print(f"  - {db_name}: {source_path}")

    outputs = run_candidate_library()
    for db_name, csv_path in outputs["per_db_csvs"].items():
        print(f"saved: {db_name} -> {csv_path}")
    print(f"saved: {outputs['stats_csv']}")
    print(f"saved: {outputs['invalid_smiles_json']}")
    print(f"saved: {outputs['run_config_json']}")
    print(f"saved: {outputs['figures']['consensus_rates']}")
    print(f"saved: {outputs['figures']['counts']}")
    print(f"saved: {outputs['figures']['ecdf']}")
