"""
Run all RUL training scripts and print a summary comparison.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_rul_baseline.py",
    "train_rul_gradient_boosting.py",
    "train_rul_extended_features.py",
    "train_rul_xgboost.py",
    "train_rul_ensemble.py",
    "train_rul_tuned.py",
]


def main():
    project_dir = Path(__file__).parent
    print("Running all RUL models...\n")

    for script in SCRIPTS:
        path = project_dir / script
        if path.exists():
            print(f"\n{'='*60}\n>>> {script}\n{'='*60}")
            result = subprocess.run([sys.executable, str(path)], cwd=project_dir)
            if result.returncode != 0:
                print(f"[!] {script} failed (e.g. missing xgboost: pip install xgboost)")
        else:
            print(f"Skipping {script} (not found)")


if __name__ == "__main__":
    main()
