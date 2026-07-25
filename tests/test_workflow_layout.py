from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPO_ROOT / "workflow"
WORKFLOWS = (
    "10_build_data",
    "20_audit_splits",
    "30_benchmark_models",
    "40_report_results",
    "50_explain_rf",
    "60_virtual_screen",
    "70_external_validation",
    "80_web_app",
)


def test_numbered_workflows_have_public_contract_files() -> None:
    for name in WORKFLOWS:
        root = WORKFLOW_ROOT / name
        assert (root / "README.md").is_file()
        assert (root / "main.py").is_file()
        assert os.access(root / "run.sh", os.X_OK)


def test_workflow_clis_resolve_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    for name in WORKFLOWS:
        completed = subprocess.run(
            [sys.executable, str(WORKFLOW_ROOT / name / "main.py"), "--help"],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.casefold()


def test_benchmark_cli_owns_its_output_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(WORKFLOW_ROOT / "30_benchmark_models" / "main.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--output-root" not in completed.stdout
    full_run = (WORKFLOW_ROOT / "30_benchmark_models" / "run_full.sh").read_text(encoding="utf-8")
    assert "gpus=(0 1 2)" in full_run
    assert "results/_logs" not in full_run  # Composed from the workflow-owned results root.


def test_reporting_cli_owns_its_input_and_output_roots() -> None:
    completed = subprocess.run(
        [sys.executable, str(WORKFLOW_ROOT / "40_report_results" / "main.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--input-root" not in completed.stdout
    assert "--output-root" not in completed.stdout
    assert "--replace" in completed.stdout


def test_explanation_cli_owns_its_inputs_and_output_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(WORKFLOW_ROOT / "50_explain_rf" / "main.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--input-root" not in completed.stdout
    assert "--output-root" not in completed.stdout
    assert "--replace" in completed.stdout


def test_build_data_keeps_datasail_provenance_out_of_dpnet_validation() -> None:
    source = (WORKFLOW_ROOT / "10_build_data" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        call.func.id: call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }

    validate_keywords = {keyword.arg for keyword in calls["validate_partitions"].keywords}
    snapshot_keywords = {keyword.arg for keyword in calls["write_runtime_snapshot"].keywords}
    assert "datasail_source" not in validate_keywords
    assert "datasail_source" in snapshot_keywords
