"""
Run the full Task 4 pipeline: prepare -> train -> generate -> evaluate.

Usage:
    python run_all.py              # full pipeline
    python run_all.py --skip-train # only prepare + generate with base model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"


def run(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SRC / script)] + (extra_args or [])
    print(f"\n{'=' * 60}\n>>> {' '.join(cmd)}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 4 pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--max-rows", type=int, default=50000, help="TSV rows to use")
    args = parser.parse_args()

    run("prepare_data.py", ["--max-rows", str(args.max_rows)])

    if not args.skip_train:
        run("train.py", ["--epochs", str(args.epochs)])

    run("generate.py")
    run("evaluate.py")

    print("\nPipeline complete. See results/ folder.")


if __name__ == "__main__":
    main()
