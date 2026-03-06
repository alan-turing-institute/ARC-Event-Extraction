"""
Figure: Comparison of argument extraction performance across different models and
evaluation regimes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from arcee.plotting import combine_recall_precision, plot_answer_matrix
from matplotlib.figure import Figure

# Load data
this_file = Path(__file__).resolve()


argument_precision_data = pd.read_csv(this_file.parent / "tuned_argument_precision.csv")
argument_recall_data = pd.read_csv(this_file.parent / "tuned_argument_recall.csv")
argument_semi_data = pd.read_csv(this_file.parent / "tuned_argument_semi_f1.csv")

tuned_map = {
    "ACE05 Labels": "ACE05 Labels",
    "Tag-Prime": "Tag-Prime",
    "llama3b512r": "Llama-3b",
    "llama8b256r": "Llama-8b",
    "mistral24b256r": "Mistral-24b",
}

##  Set up Figure ##

fig = Figure(
    figsize=(4, 7), layout="constrained"
)  # Use 'constrained' layout to optimize space
axes = fig.subplots(2, 1)

plot_answer_matrix(
    axes[0],
    combine_recall_precision(argument_recall_data, argument_precision_data),
    "Recall Argument Extraction",
    [tuned_map[label] for label in argument_precision_data.columns],
)
plot_answer_matrix(
    axes[1],
    argument_semi_data.values,
    "Partial Match Argument Extraction",
    [tuned_map[label] for label in argument_semi_data.columns],
)

# Add blank labels for space for the suplabels
for ax in axes:
    ax.set_ylabel(" ")
axes[-1].set_xlabel(" ")

fig.supxlabel("Predicted answers", y=0.005)  # Adjusted y position to bring closer
fig.supylabel("Referenced answers", x=0.005)  # Adjusted x position to bring closer

# Save the figure
fig.savefig(this_file.parent / f"{this_file.stem}.pdf")
