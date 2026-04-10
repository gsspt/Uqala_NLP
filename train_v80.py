#!/usr/bin/env python3
"""
train_v80.py
─────────────────────────────────────────────────
Simple wrapper to run v80 with the correct environment.

This script ensures you're using the correct Python version
and all dependencies are available.

Usage:
  # Direct execution (requires venv Python)
  C:\Users\augus\.conda\envs\uqala\python.exe train_v80.py [--cv 5]

  # Via shortcut (TBD)
  python train_v80.py [--cv 5]
"""

import subprocess
import sys
import pathlib
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train v80 classifier')
    parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()

    # The actual pipeline script
    pipeline_script = pathlib.Path(__file__).parent / 'pipelines' / 'level1_interpretable' / 'p1_4_logistic_regression_v80.py'

    if not pipeline_script.exists():
        print(f"Error: Pipeline script not found at {pipeline_script}")
        sys.exit(1)

    # Run it
    cmd = [sys.executable, str(pipeline_script), '--cv', str(args.cv)]
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=pipeline_script.parent.parent.parent)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
