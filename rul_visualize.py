"""Visualization utilities for RUL model evaluation."""

import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, name, save_prefix):
    """Scatter plot: actual vs predicted RUL for train and test."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    max_val = max(y_train.max(), y_test.max(), y_pred_train.max(), y_pred_test.max())

    # Train
    axes[0].scatter(y_train, y_pred_train, alpha=0.6, s=20, c="steelblue", edgecolors="navy", linewidth=0.3)
    axes[0].plot([0, max_val], [0, max_val], "k--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual RUL (cycles)")
    axes[0].set_ylabel("Predicted RUL (cycles)")
    axes[0].set_title("Train Set")
    axes[0].legend()
    axes[0].set_xlim(0, max_val * 1.05)
    axes[0].set_ylim(0, max_val * 1.05)
    axes[0].grid(True, alpha=0.3)

    # Test
    axes[1].scatter(y_test, y_pred_test, alpha=0.6, s=30, c="coral", edgecolors="darkred", linewidth=0.3)
    axes[1].plot([0, max_val], [0, max_val], "k--", lw=2, label="Perfect prediction")
    axes[1].set_xlabel("Actual RUL (cycles)")
    axes[1].set_ylabel("Predicted RUL (cycles)")
    axes[1].set_title("Test Set (B0018)")
    axes[1].legend()
    axes[1].set_xlim(0, max_val * 1.05)
    axes[1].set_ylim(0, max_val * 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{name}: Actual vs Predicted RUL", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / f"{save_prefix}_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals(y_test, y_pred_test, name, save_prefix):
    """Residual plot: residuals vs predicted for test set."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    residuals = y_test.values - y_pred_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred_test, residuals, alpha=0.6, s=30, c="teal", edgecolors="darkgreen", linewidth=0.3)
    axes[0].axhline(y=0, color="k", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted RUL (cycles)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="navy", alpha=0.8)
    axes[1].axvline(x=0, color="k", linestyle="--", lw=2)
    axes[1].set_xlabel("Residual (cycles)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{name}: Residual Analysis (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / f"{save_prefix}_residuals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_cols, name, save_prefix):
    """Bar chart of feature importance (for tree-based models)."""
    import matplotlib.pyplot as plt

    # Get feature_importances_ from model or first estimator
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # VotingRegressor: average importances from sub-estimators
        imps = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
        if imps:
            imp = np.mean(imps, axis=0)

    if imp is None:
        return

    _ensure_dir()
    idx = np.argsort(imp)[::-1]
    imp_sorted = imp[idx]
    cols_sorted = [feature_cols[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(cols_sorted)), imp_sorted, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_yticks(range(len(cols_sorted)))
    ax.set_yticklabels(cols_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{name}: Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = FIGURES_DIR / f"{save_prefix}_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rul_over_cycles(train_df, test_df, y_pred_test, name, save_prefix):
    """Line plot: actual and predicted RUL over cycle index for test battery."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    test_df = test_df.copy()
    test_df["RUL_pred"] = y_pred_test

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_df["discharge_cycle_index"], test_df["RUL"], "o-", label="Actual RUL", color="steelblue", markersize=4)
    ax.plot(test_df["discharge_cycle_index"], test_df["RUL_pred"], "s-", label="Predicted RUL", color="coral", markersize=4)
    ax.set_xlabel("Discharge Cycle Index")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{name}: RUL vs Cycle (Test Battery B0018)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"{save_prefix}_rul_over_cycles.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix_binned(y_test, y_pred_test, name, save_prefix, n_bins=3):
    """
    Confusion-matrix-like plot for regression: bin RUL into categories.
    Bins: Early (high RUL), Mid, Late (low RUL).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred_test)
    bins = np.percentile(np.concatenate([y_test_arr, y_pred_arr]), np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -0.1, 1e9

    y_test_bin = np.digitize(y_test_arr, bins[1:-1])
    y_pred_bin = np.digitize(y_pred_arr, bins[1:-1])
    labels = ["Early" if i == 0 else "Mid" if i == 1 else "Late" for i in range(n_bins)]

    cm = confusion_matrix(y_test_bin, y_pred_bin, labels=range(n_bins))

    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Count"})
    except ImportError:
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n_bins))
        ax.set_yticks(range(n_bins))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(n_bins):
            for j in range(n_bins):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax, label="Count")
    ax.set_xlabel("Predicted RUL Bin")
    ax.set_ylabel("Actual RUL Bin")
    ax.set_title(f"{name}: Binned RUL Confusion Matrix (Test Set)\nEarly=high RUL, Late=low RUL")
    plt.tight_layout()
    path = FIGURES_DIR / f"{save_prefix}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(model, X_train, y_train, X_test, y_test, train_df, test_df, feature_cols, name, save_prefix):
    """Generate all standard plots for a trained model."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\nGenerating plots...")
    plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, name, save_prefix)
    plot_residuals(y_test, y_pred_test, name, save_prefix)
    plot_feature_importance(model, feature_cols, name, save_prefix)
    plot_rul_over_cycles(train_df, test_df, y_pred_test, name, save_prefix)
    plot_confusion_matrix_binned(y_test, y_pred_test, name, save_prefix)
    print("Done.\n")
