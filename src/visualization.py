"""
Unified visualization module for anomaly detection results.

Replaces: organization_trend.py, user_trend.py

Key improvements:
- Single module for all visualization needs
- Non-blocking plot display with save-to-file option
- Configurable output formats (PNG, SVG, PDF)
- Clean API with proper figure management
- Support for both interactive and headless environments
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import get_config, OUTPUTS_DIR

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend by default
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed; visualization disabled.")


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

INTERVAL_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24"]
DAY_TYPE_COLORS = {"WD": "#2196F3", "Sat": "#FF9800", "Sun": "#F44336"}
DAY_TYPE_MARKERS = {"WD": "o", "Sat": "s", "Sun": "^"}


def _ensure_matplotlib():
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for visualization. Install with: pip install matplotlib")


# ──────────────────────────────────────────────────────────────────────
# Organization-level trends
# ──────────────────────────────────────────────────────────────────────

def plot_organization_trend(
    models: Dict[str, Dict],
    title: str = "Organization Logon Behavior",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot organization-wide logon trends across day types.

    Args:
        models: Dict mapping label -> model (e.g., {"Week 1": model1, "Week 2": model2})
        title: Plot title
        save_path: Path to save figure (None = auto-generate)
        show: Whether to display interactively

    Returns:
        Path to saved figure, or None if show=True only.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))

    line_styles = ["-", "--", "-.", ":"]

    for model_idx, (label, model) in enumerate(models.items()):
        key_list = list(model.keys())
        if not key_list:
            continue

        org_totals = {dt: np.zeros(8) for dt in ["WD", "Sat", "Sun"]}

        for user_key in key_list:
            for dt in ["WD", "Sat", "Sun"]:
                day_block = model[user_key].get(dt, {})
                ic = day_block.get("IntervalCounter", {})
                sum_vals = ic.get("sum", [0]*8)
                if isinstance(sum_vals, list):
                    org_totals[dt] += np.array(sum_vals, dtype=np.float64)

        # Normalize by day count
        sample_user = model[key_list[0]]
        for dt in ["WD", "Sat", "Sun"]:
            num_days = max(sample_user.get(dt, {}).get("DayCounter", 1), 1)
            org_totals[dt] /= num_days

        ls = line_styles[model_idx % len(line_styles)]
        for dt in ["WD", "Sat", "Sun"]:
            ax.plot(
                INTERVAL_LABELS, org_totals[dt],
                linestyle=ls,
                marker=DAY_TYPE_MARKERS[dt],
                color=DAY_TYPE_COLORS[dt],
                label=f"{label}: {dt}",
                linewidth=2,
                markersize=6,
                alpha=0.8 + 0.2 * (model_idx == 0),
            )

    ax.set_xlabel("Time Intervals (hours)", fontsize=12)
    ax.set_ylabel("Average Logons per Day", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = _save_or_show(fig, save_path, "organization_trend", show)
    return output


# ──────────────────────────────────────────────────────────────────────
# User-level trends
# ──────────────────────────────────────────────────────────────────────

def plot_user_trend(
    user: str,
    models: Dict[str, Dict],
    test_model: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot individual user logon behavior across training weeks and test data.

    Args:
        user: Username to plot
        models: Dict mapping label -> model
        test_model: Optional test model to overlay
        save_path: Path to save figure
        show: Whether to display interactively

    Returns:
        Path to saved figure.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))
    line_styles = ["-", "--", "-.", ":"]

    for model_idx, (label, model) in enumerate(models.items()):
        if user not in model:
            continue

        ls = line_styles[model_idx % len(line_styles)]
        for dt in ["WD", "Sat", "Sun"]:
            day_block = model[user].get(dt, {})
            ic = day_block.get("IntervalCounter", {})
            avg = ic.get("avg", [0]*8)
            if isinstance(avg, list) and any(v > 0 for v in avg):
                ax.plot(
                    INTERVAL_LABELS, avg,
                    linestyle=ls,
                    marker=DAY_TYPE_MARKERS[dt],
                    color=DAY_TYPE_COLORS[dt],
                    label=f"{label}: {dt}",
                    linewidth=2,
                    markersize=6,
                )

    # Overlay test data
    if test_model and user in test_model:
        for dt in ["WD"]:  # Usually test is a single day
            test_data = test_model[user].get(dt, {}).get("IntervalCounter", [0]*8)
            if isinstance(test_data, list) and any(v > 0 for v in test_data):
                ax.plot(
                    INTERVAL_LABELS, test_data,
                    linestyle="--",
                    marker="D",
                    color="#9C27B0",
                    label=f"Test: {dt}",
                    linewidth=2.5,
                    markersize=8,
                )

    ax.set_xlabel("Time Intervals (hours)", fontsize=12)
    ax.set_ylabel("Average Logons per Day", fontsize=12)
    ax.set_title(f"User {user} - Logon Behavior", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = _save_or_show(fig, save_path, f"user_trend_{user}", show)
    return output


# ──────────────────────────────────────────────────────────────────────
# Source address anomaly visualization
# ──────────────────────────────────────────────────────────────────────

def plot_source_anomalies(
    user: str,
    source_anomaly: Dict,
    baseline_model: Dict,
    test_model: Dict,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot source address anomalies for a user."""
    _ensure_matplotlib()

    if user not in source_anomaly or not source_anomaly[user]:
        logger.info("No source anomalies to plot for user %s", user)
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    sa_labels = []
    baseline_vals = []
    test_vals = []

    for sa in source_anomaly[user]:
        sa_labels.append(sa[:15])  # Truncate long IPs

        # Baseline average
        if user in baseline_model:
            sa_data = baseline_model[user].get("WD", {}).get("SourceAddress", {}).get(sa, {})
            baseline_vals.append(sa_data.get("avg", 0))
        else:
            baseline_vals.append(0)

        # Test value
        if user in test_model:
            test_vals.append(test_model[user].get("WD", {}).get("SourceAddress", {}).get(sa, 0))
        else:
            test_vals.append(0)

    x = np.arange(len(sa_labels))
    width = 0.35

    ax.bar(x - width/2, baseline_vals, width, label="Baseline Avg", color="#2196F3", alpha=0.7)
    ax.bar(x + width/2, test_vals, width, label="Test Day", color="#F44336", alpha=0.7)

    ax.set_xlabel("Source Address", fontsize=12)
    ax.set_ylabel("Logon Count", fontsize=12)
    ax.set_title(f"User {user} - Source Address Anomalies", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sa_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    output = _save_or_show(fig, save_path, f"source_anomaly_{user}", show)
    return output


# ──────────────────────────────────────────────────────────────────────
# Risk score distribution
# ──────────────────────────────────────────────────────────────────────

def plot_risk_distribution(
    risk_scores: Dict[str, float],
    thresholds: List[float],
    title: str = "Risk Score Distribution",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot distribution of risk scores with threshold lines."""
    _ensure_matplotlib()

    scores = list(risk_scores.values())
    if not scores:
        logger.info("No risk scores to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(scores, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")

    if len(thresholds) >= 2:
        ax.axvline(thresholds[0], color="#FF9800", linestyle="--", linewidth=2,
                   label=f"Lower threshold ({thresholds[0]:.1f})")
        ax.axvline(thresholds[1], color="#F44336", linestyle="--", linewidth=2,
                   label=f"Upper threshold ({thresholds[1]:.1f})")

    ax.set_xlabel("Risk Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    output = _save_or_show(fig, save_path, "risk_distribution", show)
    return output


# ──────────────────────────────────────────────────────────────────────
# RL training progress
# ──────────────────────────────────────────────────────────────────────

def plot_rl_training(
    losses: List[float],
    rewards: List[float],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot RL agent training progress."""
    _ensure_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if losses:
        ax1.plot(losses, color="#2196F3", alpha=0.5, linewidth=0.5)
        # Moving average
        window = min(50, len(losses) // 5 + 1)
        if window > 1:
            ma = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax1.plot(range(window-1, len(losses)), ma, color="#F44336", linewidth=2, label=f"MA({window})")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("DQN Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Reward curve
    if rewards:
        ax2.plot(rewards, color="#4CAF50", alpha=0.5, linewidth=0.5)
        window = min(50, len(rewards) // 5 + 1)
        if window > 1:
            ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax2.plot(range(window-1, len(rewards)), ma, color="#FF9800", linewidth=2, label=f"MA({window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_title("RL Agent Rewards")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.suptitle("RL Threshold Optimizer Training", fontsize=14, fontweight="bold")
    fig.tight_layout()

    output = _save_or_show(fig, save_path, "rl_training", show)
    return output


# ──────────────────────────────────────────────────────────────────────
# Detection summary dashboard
# ──────────────────────────────────────────────────────────────────────

def plot_detection_summary(
    time_anomalies: int,
    source_anomalies: int,
    dest_anomalies: int,
    new_users: int,
    total_users: int,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot a summary dashboard of detection results."""
    _ensure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of anomaly counts
    categories = ["Time", "Source", "Destination", "New Users"]
    counts = [time_anomalies, source_anomalies, dest_anomalies, new_users]
    colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0"]

    axes[0].bar(categories, counts, color=colors, alpha=0.8, edgecolor="white")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Anomalies by Type")
    axes[0].grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Pie chart: anomalous vs normal
    total_anomalous = time_anomalies + new_users
    normal = max(total_users - total_anomalous, 0)
    sizes = [total_anomalous, normal]
    labels = [f"Anomalous ({total_anomalous})", f"Normal ({normal})"]
    pie_colors = ["#F44336", "#4CAF50"]

    axes[1].pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
    axes[1].set_title("User Classification")

    fig.suptitle("Detection Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()

    output = _save_or_show(fig, save_path, "detection_summary", show)
    return output


# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────

def _save_or_show(
    fig, save_path: Optional[Path], default_name: str, show: bool
) -> Optional[Path]:
    """Save figure and/or display it."""
    output = None
    if save_path is None and not show:
        save_path = OUTPUTS_DIR / f"{default_name}.png"

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
        output = save_path

    if show:
        plt.show()

    plt.close(fig)
    return output
