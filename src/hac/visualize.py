"""General visualization helpers for HAC experiments."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


def plot_lightcurve_comparison(
    target_curves: ArrayLike,
    initial_guess_curves: ArrayLike,
    optimized_curves: ArrayLike,
    observer_labels: Sequence[str],
    curve_labels: tuple[str, str, str],
    output_path: str | PathLike[str],
    *,
    title: str | None = None,
    column_count: int = 4,
    dpi: int = 160,
) -> None:
    """Plot target, initial-guess, and optimized light curves for every observer."""
    target = _curve_matrix("target_curves", target_curves)
    initial_guess = _curve_matrix("initial_guess_curves", initial_guess_curves)
    optimized = _curve_matrix("optimized_curves", optimized_curves)
    if initial_guess.shape != target.shape or optimized.shape != target.shape:
        raise ValueError(
            "target_curves, initial_guess_curves, and optimized_curves must have the "
            f"same shape; got {target.shape}, {initial_guess.shape}, and "
            f"{optimized.shape}."
        )

    observer_count = target.shape[0]
    if len(observer_labels) != observer_count:
        raise ValueError(
            "observer_labels must contain one label per curve; "
            f"got {len(observer_labels)} labels for {observer_count} curves."
        )
    if len(curve_labels) != 3:
        raise ValueError(f"curve_labels must contain three labels; got {curve_labels}.")
    if column_count <= 0:
        raise ValueError(f"column_count must be positive; got {column_count}.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = (observer_count + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(14, 2.6 * row_count),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.flat
    for observer_index, observer_label in enumerate(observer_labels):
        ax = flat_axes[observer_index]
        ax.plot(
            target[observer_index],
            label=curve_labels[0],
            linewidth=2,
        )
        ax.plot(initial_guess[observer_index], label=curve_labels[1])
        ax.plot(optimized[observer_index], label=curve_labels[2])
        ax.set_title(observer_label)
        ax.grid(alpha=0.2)
    for ax in flat_axes[observer_count:]:
        ax.set_visible(False)

    fig.supxlabel("Optimization phase sample")
    fig.supylabel("Normalized intensity")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    if title is not None:
        fig.suptitle(title, y=0.995)
    top = 0.965 if title is not None else 0.98
    fig.tight_layout(rect=(0.02, 0.02, 1.0, top))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_optimization_loss(
    loss: ArrayLike,
    output_path: str | PathLike[str],
    *,
    reference_loss: float | None = None,
    reference_label: str | None = None,
    title: str = "Large Steps optimization loss",
    dpi: int = 160,
) -> None:
    """Plot optimization loss and an optional constant reference loss."""
    values = _loss_vector("loss", loss)
    reference_value = None
    if reference_loss is not None:
        reference_value = float(reference_loss)
        if not np.isfinite(reference_value) or reference_value < 0.0:
            raise ValueError(
                "reference_loss must be a finite, non-negative value; "
                f"got {reference_loss}."
            )
        if not reference_label:
            raise ValueError("reference_label is required when reference_loss is set.")
    elif reference_label is not None:
        raise ValueError("reference_loss is required when reference_label is set.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iterations, values, label="Curve MSE")
    if reference_value is not None:
        ax.axhline(
            reference_value,
            label=reference_label,
            linestyle="--",
            color="black",
        )
    ax.set(
        xlabel="Iteration",
        ylabel="Curve MSE",
        yscale="log",
        title=title,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


VOXEL_MEASURE_LABELS: tuple[str, str] = (
    "Measure 1: 1 - |A∩B| / |A∪B|",
    "Measure 2: (|A\\B| + |B\\A|) / (|A| + |B|)",
)


def plot_voxel_measures(
    iterations: ArrayLike,
    measure1: ArrayLike,
    measure2: ArrayLike,
    output_path: str | PathLike[str],
    *,
    title: str = "Voxel volume difference against the reference shape",
    dpi: int = 160,
) -> None:
    """Plot both FIPS voxel volume-difference measures against the iteration count.

    The minimum of each curve is marked and annotated: lower is closer to the
    reference shape, and the best-scoring iteration is rarely the last one.
    """
    steps = _loss_vector("iterations", iterations)
    first = _loss_vector("measure1", measure1)
    second = _loss_vector("measure2", measure2)
    if first.shape != steps.shape or second.shape != steps.shape:
        raise ValueError(
            "iterations, measure1, and measure2 must have the same length; "
            f"got {steps.shape}, {first.shape}, and {second.shape}."
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for values, label, color in zip(
        (first, second),
        VOXEL_MEASURE_LABELS,
        ("tab:blue", "tab:orange"),
        strict=True,
    ):
        ax.plot(steps, values, label=label, color=color, marker=".", markersize=4)
        best_index = int(np.argmin(values))
        ax.scatter(
            steps[best_index],
            values[best_index],
            color=color,
            marker="*",
            s=140,
            zorder=3,
            label=(
                f"best {values[best_index]:.4f} @ iteration {int(steps[best_index])}"
            ),
        )
    ax.set(
        xlabel="Iteration",
        ylabel="Relative volume difference",
        title=title,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _curve_matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(
            f"{name} must have shape (observer, phase) with positive dimensions; "
            f"got {array.shape}."
        )
    return array


def _loss_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array
