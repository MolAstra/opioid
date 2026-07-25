from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from opioid.evaluation.dpbench import PROTOCOLS
from opioid.evaluation.split_audit import (
    REQUIRED_ANALYSIS_FILES,
    load_dpnet_split_audit,
    run_dpnet_split_audit,
    write_split_audit_index,
    write_test_development_proximity_figure,
)


def _report(protocol: str, *, shared_scaffolds: int = 0) -> dict[str, object]:
    return {
        "kind": "dpnet.split_analysis",
        "schema_version": 2,
        "task_id": "muor_antagonism",
        "processor": {"package_version": "0.15.0"},
        "source": {
            "processed_dir": f"processed_{protocol}",
            "method": protocol,
            "layout": "cv",
        },
        "split_summary": [
            {"group": "development", "rows": 884},
            {"group": "test", "rows": 98},
        ],
        "pairwise_similarity": [
            {
                "left_group": "development",
                "right_group": "test",
                "pair_count": 86632,
                "mean": 0.17,
                "std": 0.08,
                "min": 0.01,
                "max": 1.0,
                "left_nearest_mean": 0.50,
                "right_nearest_mean": 0.69,
            }
        ],
        "scaffolds": {
            "overlap": [
                {
                    "left_group": "development",
                    "right_group": "test",
                    "shared_nonempty_scaffolds": shared_scaffolds,
                }
            ]
        },
        "similarity": {
            "available": True,
            "protocol": {
                "fingerprint": "Morgan / ECFP4",
                "radius": 2,
                "n_bits": 1024,
                "similarity": "Tanimoto",
            },
            "datasail_compatible_leakage": {
                "scaled_dl": 0.15,
                "total_dl": 10.0,
                "max_dl": 60.0,
            },
        },
        "datasail": {"available": protocol == "datasail"},
    }


def _write_analysis(path: Path, protocol: str, *, shared_scaffolds: int = 0) -> None:
    path.mkdir(parents=True)
    for name in REQUIRED_ANALYSIS_FILES:
        (path / name).write_text("", encoding="utf-8")
    (path / "report.json").write_text(
        json.dumps(_report(protocol, shared_scaffolds=shared_scaffolds)),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "kind": "nearest_cross_group",
                "left_group": "development",
                "right_group": "test",
                "direction": "right_to_left",
                "bin_lower": 0.0,
                "bin_upper": 0.5,
                "count": 40,
            },
            {
                "kind": "nearest_cross_group",
                "left_group": "development",
                "right_group": "test",
                "direction": "right_to_left",
                "bin_lower": 0.5,
                "bin_upper": 1.0,
                "count": 58,
            },
        ]
    ).to_csv(path / "similarity_histogram.csv", index=False)


def test_run_dpnet_split_audit_uses_official_analyze_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "analysis"
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        _write_analysis(output, "random")

    monkeypatch.setattr("opioid.evaluation.split_audit.subprocess.run", fake_run)
    result = run_dpnet_split_audit(
        protocol="random",
        split_root=tmp_path / "dpbench",
        output_dir=output,
        dpnet_executable="dpnet-test",
        replace=True,
    )

    assert result.report_html == output / "report.html"
    assert commands == [
        [
            "dpnet-test",
            "analyze",
            "muor_antagonism",
            "--root",
            str((tmp_path / "dpbench").resolve()),
            "--processed-dir",
            "processed_random",
            "--output",
            str(output.resolve()),
            "--force",
        ]
    ]


def test_scaffold_isolation_is_enforced_from_dpnet_report(tmp_path: Path) -> None:
    output = tmp_path / "analysis"
    _write_analysis(output, "scaffold", shared_scaffolds=1)

    with pytest.raises(AssertionError, match="DPNet report"):
        load_dpnet_split_audit(protocol="scaffold", output_dir=output)


def test_cross_protocol_index_uses_official_metrics(tmp_path: Path) -> None:
    results = []
    for protocol in PROTOCOLS:
        output = tmp_path / protocol
        _write_analysis(output, protocol)
        results.append(load_dpnet_split_audit(protocol=protocol, output_dir=output))

    paths = write_split_audit_index(results, tmp_path / "summary")
    summary = pd.read_csv(paths["summary_csv"])

    assert set(summary["protocol"]) == set(PROTOCOLS)
    assert set(summary["dpnet_version"]) == {"0.15.0"}
    assert summary.loc[summary["protocol"] == "datasail", "datasail_adapter_diagnostics"].item()


def test_cross_protocol_proximity_figure_uses_dpnet_histograms(tmp_path: Path) -> None:
    results = []
    for protocol in PROTOCOLS:
        output = tmp_path / protocol
        _write_analysis(output, protocol)
        results.append(load_dpnet_split_audit(protocol=protocol, output_dir=output))

    figure = write_test_development_proximity_figure(results, tmp_path / "figures")

    assert figure.name == "test_to_development_max_tanimoto.png"
    assert figure.is_file()
    assert figure.stat().st_size > 0
