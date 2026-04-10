#!/usr/bin/env python3
"""
run_with_venv.py
────────────────────────────────────────────────────────
Wrapper script: runs Python scripts with the correct venv.

This script activates the conda environment 'uqala' before running
any pipeline or analysis script, ensuring CAMeL Tools and all
dependencies are available.

Usage:
  python3 run_with_venv.py <script> [args...]

Examples:
  python3 run_with_venv.py pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5
  python3 run_with_venv.py scripts/analyze_corpus.py
"""

import sys
import subprocess
import pathlib
import os

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Path to the conda environment
VENV_PYTHON = r"C:\Users\augus\.conda\envs\uqala\python.exe"

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_with_venv.py <script> [args...]")
        print("\nExample:")
        print("  python run_with_venv.py pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5")
        sys.exit(1)

    script = sys.argv[1]
    args = sys.argv[2:]

    # Verify the script exists (absolute path)
    script_path = pathlib.Path(script).resolve()
    if not script_path.exists():
        print(f"Script not found: {script}")
        sys.exit(1)

    # Get the repo root (parent of pipelines, scripts, etc.)
    repo_root = pathlib.Path(__file__).parent.resolve()

    # Build command
    cmd = [VENV_PYTHON, str(script_path)] + args

    print(f"Running with venv: {VENV_PYTHON}")
    print(f"Script: {script}")
    if args:
        print(f"Args: {' '.join(args)}")
    print("=" * 80)

    # Run the script in the repo root
    try:
        result = subprocess.run(cmd, cwd=repo_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error running script: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
