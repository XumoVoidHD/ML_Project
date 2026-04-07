"""
Run all RUL training scripts and print a summary comparison.
"""

import subprocess
import sys

MODULES = [
    "scripts.training.train_rul_baseline",
    "scripts.training.train_rul_gradient_boosting",
    "scripts.training.train_rul_extended_features",
    "scripts.training.train_rul_xgboost",
    "scripts.training.train_rul_ensemble",
    "scripts.training.train_rul_tuned",
]


def main():
    print("Running all RUL models...\n")

    for module_name in MODULES:
        print(f"\n{'='*60}\n>>> {module_name}\n{'='*60}")
        result = subprocess.run([sys.executable, "-m", module_name])
        if result.returncode != 0:
            print(f"[!] {module_name} failed (e.g. missing xgboost: pip install xgboost)")


if __name__ == "__main__":
    main()
