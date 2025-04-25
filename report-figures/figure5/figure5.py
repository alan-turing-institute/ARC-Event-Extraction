"""
Figure 5: Comparison of the Answers
"""

from __future__ import annotations

import os

import colorcet as cc
import numpy as np
import pandas as pd
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure


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


# Load data
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FINE TUNING EAE DATA ##

example_event_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure5/example_event_f1.csv")
)
example_argument_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure5/example_argument_f1.csv")
)
zero_argument_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure5/zero_argument_f1.csv")
)
tuned_argument_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure5/tuned_argument_f1.csv")
)

tuned_map = {
    "ACE05 Labels": "ACE05 Labels",
    "llama3b512r": "Llama-3b",
    "llama8b256r": "Llama-8b",
    "mistral24b256r": "Mistral-24b",
}

##  Set up Figure ##

fig = Figure(
    figsize=(8, 8), layout="constrained"
)  # Use 'constrained' layout to optimize space
axes = fig.subplots(2, 2)

plot_answer_matrix(
    axes[0, 0],
    example_event_data.values,
    "Few-shot Event Detection",
    example_event_data.columns.tolist(),
)
plot_answer_matrix(
    axes[0, 1],
    example_argument_data.values,
    "Few-shot Argument Extraction",
    example_argument_data.columns.tolist(),
)
plot_answer_matrix(
    axes[1, 0],
    zero_argument_data.values,
    "Zero-shot Argument Extraction",
    zero_argument_data.columns.tolist(),
)
plot_answer_matrix(
    axes[1, 1],
    tuned_argument_data.values,
    "Fine-Tuned Argument Extraction",
    [tuned_map[label] for label in tuned_argument_data.columns],
)

fig.supxlabel("Predicted answers", y=0.005)  # Adjusted y position to bring closer
fig.supylabel("Referenced answers", x=0.005)  # Adjusted x position to bring closer

# Save the figure
fig.savefig(os.path.join(main_dir, "report-figures/figure5/figure5.pdf"))
