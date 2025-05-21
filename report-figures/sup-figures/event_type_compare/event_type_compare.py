# Appendix Figure Event Type Comparison

from __future__ import annotations

import os

import colorcet as cc
import numpy as np
import pandas as pd
from matplotlib.axes import Axes as Axis
from matplotlib.figure import Figure


def combine_recall_precision(
    recall: pd.DataFrame,
    precision: pd.DataFrame,
) -> np.ndarray:
    """
    Combine recall and precision data into a single matrix.
    Where the upper triangle is recall and the lower triangle is precision.
    This takes the upper right triangle from both the recall and precision dataframes
    and combines them into a single matrix.

    Args:
        recall (pd.DataFrame): The recall data.
        precision (pd.DataFrame): The precision data.

    Returns:
        numpy.ndarray: The combined matrix to hand to the plot function.
    """
    combined_matrix = np.zeros((len(recall), len(recall)))
    for i, j in np.ndindex(recall.shape):
        if i <= j:
            combined_matrix[i, j] = recall.iloc[i, j]
        elif i > j:
            combined_matrix[i, j] = precision.iloc[j, i]
    return combined_matrix


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
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

## Recall Precision Data ##

example_type_f1 = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/event_type_compare/example_event_type_f1.csv",
    )
)
example_type_recall = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/event_type_compare/example_event_type_recall.csv",
    )
)
example_type_precision = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/event_type_compare/example_event_type_precision.csv",
    )
)

##  Set up Figure ##

fig = Figure(figsize=(8, 4), layout="tight")
axes = fig.subplots(1, 2)

plot_answer_matrix(
    axes[0],
    example_type_f1.values,
    "Event Type F1 Score",
    example_type_f1.columns.tolist(),
)
plot_answer_matrix(
    axes[1],
    combine_recall_precision(example_type_recall, example_type_precision),
    "Event Type Recall",
    example_type_recall.columns.tolist(),
)

fig.supxlabel("Predicted answers")
fig.supylabel("Referenced answers")

# Save the figure
fig.savefig(
    os.path.join(
        main_dir, "report-figures/sup-figures/event_type_compare/event_type_compare.pdf"
    )
)
