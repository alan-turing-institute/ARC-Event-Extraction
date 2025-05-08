"""
RAMS Consider: Panel A () and Panel B (Fine Tuning)
"""

from __future__ import annotations

import os

import colorcet as cc
import numpy as np
import pandas as pd
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator


def plot_answer_matrix(
    axis: Axis,
    matrix: np.ndarray,
    title: str,
    model_names: list[str],
) -> None:
    """
    Plot the confusion matrix based on the given confusion values and source names. The
    figure is shown and also saved to the given file name.

    Args:
        axis (Axis): The axis to plot on.
        matrix (np.ndarray): The confusion matrix.
        title (str): The title of the plot.
        model_names (list[str]): The names of the models.
    """
    number_of_classes = len(model_names)
    if model_names[0] == "ACE05 Labels":
        model_names[0] = "ACE05\nLabels"
    # Plotting the matrix
    axis.matshow(matrix.astype(float), cmap=cc.cm.kbc_r)

    axis.set_title(f"{title}")

    # Adding text in each cell
    for (i, j), val in np.ndenumerate(matrix):
        axis.text(j, i, f"{100*val:.0f}", ha="center", va="center", color="black")

    # Setting tick marks for models
    axis.set_xticks(np.arange(number_of_classes))
    axis.set_yticks(np.arange(number_of_classes))
    axis.set_xticklabels(model_names, rotation=40)
    axis.xaxis.set_ticks_position("bottom")
    axis.set_yticklabels(model_names)


# Set top-level directory
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

# Load data for Panel A

rams_f1 = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/rams_consider/rams_example_argument_f1.csv",
    )
)

# Load data for Panel B
ft_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/rams_consider/rams_mistral24b_tune_eae.csv",
    )
)

# Convert Fraction to Percentage
ft_data["argument_f1"] = ft_data["argument_f1"] * 100
ft_data["bs_std"] = ft_data["bs_std"] * 100

# Filter data for Panel B
base_model = ft_data[~ft_data["fine_tune"]]
ft_model = ft_data[ft_data["fine_tune"]]

## Set up Figure ##

fig = Figure(figsize=(8, 4.5), layout="tight")
axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.1]})

ax1 = axes[0]
ax2 = axes[1]

## PANEL A ##

plot_answer_matrix(
    axes[0],
    rams_f1.values,
    "RAMS F1 Score",
    rams_f1.columns.tolist(),
)

## PANEL B ##

ax2.errorbar(
    ft_model["examples"],
    ft_model["argument_f1"],
    yerr=ft_model["bs_std"],
    fmt="D",
    label="Fine Tuned",
    color="#FF7F0E",
    capsize=8,
    linestyle="--",
)
ax2.errorbar(
    base_model["examples"],
    base_model["argument_f1"],
    yerr=base_model["bs_std"],
    fmt="8",
    label="Base Model",
    color="#1F77B4",
    capsize=8,
    linestyle="--",
)

ax2.text(
    0.03,
    0.97,
    "Fine Tuning on ACE05\nEvaluate on RAMS\nMistral 24B with similar examples",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)

ax2.set_ylabel("Argument F1")
ax2.set_xlabel("Number of examples")
ax2.set_ylim(23, 51)
ax2.grid(axis="y")
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(axis="y", direction="in", which="both", left=True, right=True)
ax2.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.grid(axis="x")
ax2.legend(loc="lower right")

# Add subplot labels
ax1.text(
    -0.1,
    1.1,
    "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax2.text(
    -0.08,
    1.0,
    "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)

# Save the figure
fig.savefig(
    os.path.join(main_dir, "report-figures/sup-figures/rams_consider/rams_consider.pdf")
)
