#!/usr/bin/env python3
"""Launch the Workflow 80 LitServe/FastAPI μORScreen application."""

from __future__ import annotations

import argparse
from pathlib import Path

from opioid.web.litserve_app import create_server


WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Workflow 60 three-model consensus and RF explanation website."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="warning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    server = create_server(
        repo_root=REPO_ROOT,
        frontend_root=WORKFLOW_DIR / "frontend",
    )
    server.run(
        host=args.host,
        port=args.port,
        num_api_servers=1,
        log_level=args.log_level,
        api_server_worker_type="process",
        pretty_logs=True,
        generate_client_file=False,
    )


if __name__ == "__main__":
    main()
