# Appendix Figure Recall Precision

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

example_event_recall = pd.read_csv(
    os.path.join(
        main_dir, "report-figures/sup-figures/recall_compare/example_event_recall.csv"
    )
)
example_event_precision = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/recall_compare/example_event_precision.csv",
    )
)
example_argument_recall = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/recall_compare/example_argument_recall.csv",
    )
)
example_argument_precision = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/recall_compare/example_argument_precision.csv",
    )
)
zero_argument_recall = pd.read_csv(
    os.path.join(
        main_dir, "report-figures/sup-figures/recall_compare/zero_argument_recall.csv"
    )
)
zero_argument_precision = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/recall_compare/zero_argument_precision.csv",
    )
)
tuned_argument_recall = pd.read_csv(
    os.path.join(
        main_dir, "report-figures/sup-figures/recall_compare/tuned_argument_recall.csv"
    )
)
tuned_argument_precision = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/recall_compare/tuned_argument_precision.csv",
    )
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
    combine_recall_precision(example_event_recall, example_event_precision),
    "Few-shot Event Detection",
    example_event_recall.columns.tolist(),
)
plot_answer_matrix(
    axes[0, 1],
    combine_recall_precision(example_argument_recall, example_argument_precision),
    "Few-shot Argument Extraction",
    example_argument_recall.columns.tolist(),
)
plot_answer_matrix(
    axes[1, 0],
    combine_recall_precision(zero_argument_recall, zero_argument_precision),
    "Zero-shot Argument Extraction",
    zero_argument_recall.columns.tolist(),
)
plot_answer_matrix(
    axes[1, 1],
    combine_recall_precision(tuned_argument_recall, tuned_argument_precision),
    "Fine-Tuned Argument Extraction",
    [tuned_map[label] for label in tuned_argument_recall.columns],
)

fig.supxlabel("Predicted answers", y=0.005)  # Adjusted y position to bring closer
fig.supylabel("Referenced answers", x=0.005)  # Adjusted x position to bring closer

# Save the figure
fig.savefig(
    os.path.join(
        main_dir, "report-figures/sup-figures/recall_compare/recall_compare.pdf"
    )
)
