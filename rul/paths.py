"""Shared filesystem paths for the battery RUL project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed_rul"
RAW_CYCLES_DIR = DATA_DIR / "data"
COMBINED_DATA_DIR = DATA_DIR / "combined_by_battery"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
FIGURES_DIR = PROJECT_DIR / "figures"
