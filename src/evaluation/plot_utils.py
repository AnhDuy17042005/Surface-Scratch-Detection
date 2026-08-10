"""
    Shared plotting helpers for evaluation scripts.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def load_pyplot():
    """
        Import matplotlib lazily so metric-only imports stay lightweight.
    """

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def safe_float(value: Any, default: float = 0.0) -> float:
    """
        Convert a value to float and fall back for missing/blank CSV fields.
    """

    if value in (None, ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def smooth_values(values: list[float], window: int = 5) -> list[float]:
    """
        Smooth plot values with a centered moving average.
    """

    if len(values) < 3:
        return values

    window = max(1, min(window, len(values)))
    half_window = window // 2
    smoothed: list[float] = []

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        smoothed.append(float(np.mean(values[start:end])))

    return smoothed


def save_binary_confusion_matrix_plot(
    counts: dict[str, int],
    output_path: Path,
    class_names: tuple[str, str],
) -> None:
    """
        Save a YOLO-style normalized binary confusion matrix.
    """

    plt = load_pyplot()

    """Rows are predicted classes, columns are true classes."""
    matrix = np.array(
        [
            [counts["tp"], counts["fp"]],
            [counts["fn"], counts["tn"]],
        ],
        dtype=np.float64,
    )

    """Ultralytics normalizes confusion matrices by true-class columns."""
    column_sums = matrix.sum(axis=0, keepdims=True)
    display = np.divide(
        matrix,
        column_sums,
        out=np.zeros_like(matrix),
        where=column_sums > 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    image = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix Normalized")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xticks([0, 1], labels=list(class_names))
    ax.set_yticks([0, 1], labels=list(class_names))
    ax.tick_params(axis="x", labelrotation=90)

    for row_index in range(2):
        for col_index in range(2):
            value = display[row_index, col_index]
            if value <= 0:
                continue

            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.45 else "#111827",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_global_metrics_bar(
    metrics: dict[str, Any],
    output_path: Path,
    metric_names: list[str],
    title: str,
) -> None:
    """
        Save a global metrics bar chart.
    """

    plt = load_pyplot()
    values = [safe_float(metrics.get(name)) for name in metric_names]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bars = ax.bar(
        [name.replace("_", "\n").replace("map", "mAP").title() for name in metric_names],
        values,
        color=["#2563eb", "#16a34a", "#7c3aed", "#f59e0b", "#dc2626", "#0891b2"],
    )
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.04)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.08),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_threshold_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    x_name: str,
    metric_names: list[str],
    title: str,
) -> None:
    """
        Save metric curves over a threshold-like x-axis.
    """

    if not rows:
        return

    plt = load_pyplot()
    xs = [safe_float(row.get(x_name)) for row in rows]
    colors = ["#f59e0b", "#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#0891b2"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for metric_name, color in zip(metric_names, colors):
        values = [safe_float(row.get(metric_name)) for row in rows]
        ax.plot(
            xs,
            values,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=color,
            label=metric_name.replace("_", " ").title(),
        )

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(x_name.replace("_", " ").title())
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_yolo_results_csv(path: Path) -> list[dict[str, float]]:
    """
        Load Ultralytics results.csv rows with stripped column names.
    """

    if not path.is_file():
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append({
                key.strip(): safe_float(value)
                for key, value in row.items()
                if key is not None
            })
        return rows


def resolve_yolo_history_path(model_path: Path, history_path: Path | None = None) -> Path | None:
    """
        Find Ultralytics results.csv for a best.pt checkpoint.
    """

    if history_path is not None:
        return history_path.resolve()

    candidates = [
        model_path.parent.parent / "results.csv",
        model_path.parent / "results.csv",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def save_yolo_training_history(
    rows: list[dict[str, float]],
    output_path: Path,
    task: str,
) -> None:
    """
        Save YOLO training history in an 8-panel layout.
    """

    if not rows:
        return

    plt = load_pyplot()
    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(rows)]

    if task == "segment":
        specs = [
            ("train/box_loss", "train/box_loss"),
            ("train/seg_loss", "train/seg_loss"),
            ("train/cls_loss", "train/cls_loss"),
            ("val/box_loss", "val/box_loss"),
            ("val/seg_loss", "val/seg_loss"),
            ("metrics/precision(M)", "metrics/precision(M)"),
            ("metrics/recall(M)", "metrics/recall(M)"),
            ("metrics/mAP50(M)", "metrics/mAP50(M)"),
        ]
    else:
        specs = [
            ("train/box_loss", "train/box_loss"),
            ("train/cls_loss", "train/cls_loss"),
            ("train/dfl_loss", "train/dfl_loss"),
            ("val/box_loss", "val/box_loss"),
            ("val/cls_loss", "val/cls_loss"),
            ("metrics/precision(B)", "metrics/precision(B)"),
            ("metrics/recall(B)", "metrics/recall(B)"),
            ("metrics/mAP50(B)", "metrics/mAP50(B)"),
        ]

    available_specs = [
        (title, key)
        for title, key in specs
        if any(key in row for row in rows)
    ]
    if not available_specs:
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=150)
    axes = axes.ravel()

    for axis in axes:
        axis.axis("off")

    for axis, (title, key) in zip(axes, available_specs):
        axis.axis("on")
        values = [safe_float(row.get(key)) for row in rows]
        axis.plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label="results",
        )
        axis.plot(
            epochs,
            smooth_values(values),
            linestyle=":",
            linewidth=2.0,
            color="#ff7f0e",
            label="smooth",
        )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)

    axes[min(1, len(available_specs) - 1)].legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_image_collage(
    source_paths: list[Path],
    output_path: Path,
    title: str,
) -> bool:
    """
        Save a collage from existing Ultralytics plot images.
    """

    existing = [path for path in source_paths if path.is_file()]
    if not existing:
        return False

    if len(existing) == 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing[0], output_path)
        return True

    plt = load_pyplot()
    cols = 2
    rows = int(np.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), dpi=150)
    axes_array = np.array(axes).reshape(-1)

    for axis in axes_array:
        axis.axis("off")

    for axis, image_path in zip(axes_array, existing):
        image = plt.imread(str(image_path))
        axis.imshow(image)
        axis.set_title(image_path.stem)
        axis.axis("off")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True
