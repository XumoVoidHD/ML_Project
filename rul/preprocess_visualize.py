"""
Visualizations for preprocessed RUL data (EDA).
Run after preprocess_rul.py to generate exploratory plots.
"""

import numpy as np
import pandas as pd

from rul.paths import FIGURES_DIR, PREPROCESSED_DIR


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_capacity_degradation(df, save_path):
    """Capacity vs cycle index per battery (degradation curves)."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"B0005": "steelblue", "B0006": "coral", "B0007": "seagreen", "B0018": "mediumpurple"}
    for bid in df["battery_id"].unique():
        b = df[df["battery_id"] == bid].sort_values("discharge_cycle_index")
        ax.plot(b["discharge_cycle_index"], b["Capacity"], "o-", label=bid, color=colors.get(bid, "gray"), markersize=3)

    ax.axhline(y=1.4, color="red", linestyle="--", lw=2, label="EOL (1.4 Ah)")
    ax.set_xlabel("Discharge Cycle Index")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Capacity Degradation Curves by Battery")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rul_distribution(df, save_path):
    """RUL distribution (train vs test)."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    train = df[df["battery_id"].isin(["B0005", "B0006", "B0007"])]
    test = df[df["battery_id"] == "B0018"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(train["RUL"], bins=25, color="steelblue", edgecolor="navy", alpha=0.8, label="Train")
    axes[0].set_xlabel("RUL (cycles)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RUL Distribution (Train: B0005, B0006, B0007)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(test["RUL"], bins=20, color="coral", edgecolor="darkred", alpha=0.8, label="Test (B0018)")
    axes[1].set_xlabel("RUL (cycles)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("RUL Distribution (Test: B0018)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("RUL Target Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_heatmap(df, save_path):
    """Feature correlation heatmap."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    feat_cols = [
        "Capacity", "capacity_fade", "Re", "Rct", "discharge_duration",
        "avg_temperature", "voltage_at_100s", "voltage_at_300s", "voltage_at_600s",
        "discharge_cycle_index", "RUL",
    ]
    cols = [c for c in feat_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    plt.colorbar(im, ax=ax, label="Correlation")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_distributions(df, save_path):
    """Histograms of key features."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    cols = ["Capacity", "capacity_fade", "Re", "Rct", "discharge_duration", "RUL"]
    cols = [c for c in cols if c in df.columns]

    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_2d(axes)

    for idx, col in enumerate(cols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="navy", alpha=0.8)
        axes[r, c].set_xlabel(col)
        axes[r, c].set_ylabel("Count")
        axes[r, c].set_title(f"{col} Distribution")
        axes[r, c].grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].axis("off")

    fig.suptitle("Feature Distributions (Preprocessed Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_impedance_trends(df, save_path):
    """Re and Rct over cycle index per battery."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for bid in df["battery_id"].unique():
        b = df[df["battery_id"] == bid].sort_values("discharge_cycle_index")
        axes[0].plot(b["discharge_cycle_index"], b["Re"], "o-", label=bid, markersize=2)
        axes[1].plot(b["discharge_cycle_index"], b["Rct"], "o-", label=bid, markersize=2)

    axes[0].set_xlabel("Discharge Cycle Index")
    axes[0].set_ylabel("Re (Ω)")
    axes[0].set_title("Electrolyte Resistance (Re) over Cycles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Discharge Cycle Index")
    axes[1].set_ylabel("Rct (Ω)")
    axes[1].set_title("Charge Transfer Resistance (Rct) over Cycles")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Impedance Trends by Battery", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all():
    """Generate all preprocessing visualizations."""
    path = PREPROCESSED_DIR / "rul_preprocessed.csv"
    if not path.exists():
        print("Run preprocess_rul.py first to generate rul_preprocessed.csv")
        return

    df = pd.read_csv(path)
    print("\nGenerating preprocessing visualizations...")

    plot_capacity_degradation(df, FIGURES_DIR / "preprocess_capacity_degradation.png")
    plot_rul_distribution(df, FIGURES_DIR / "preprocess_rul_distribution.png")
    plot_correlation_heatmap(df, FIGURES_DIR / "preprocess_correlation_heatmap.png")
    plot_feature_distributions(df, FIGURES_DIR / "preprocess_feature_distributions.png")
    plot_impedance_trends(df, FIGURES_DIR / "preprocess_impedance_trends.png")

    print("Done.\n")


if __name__ == "__main__":
    generate_all()
