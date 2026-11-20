"""
Figure: Comparison of argument extraction performance across different models and
evaluation regimes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from arcee.plotting import plot_answer_matrix
from matplotlib.figure import Figure

# Load data
this_file = Path(__file__).resolve()

zero_argument_data = pd.read_csv(this_file.parent / "zero_argument_f1.csv")
example_argument_data = pd.read_csv(this_file.parent / "example_argument_f1.csv")
tuned_argument_data = pd.read_csv(this_file.parent / "tuned_argument_f1.csv")

zero_event_data = pd.read_csv(this_file.parent / "zero_event_f1.csv")
example_event_data = pd.read_csv(this_file.parent / "example_event_f1.csv")
example_event_type_data = pd.read_csv(this_file.parent / "example_event_type_f1.csv")

tuned_map = {
    "ACE05 Labels": "ACE05 Labels",
    "Tag-Prime": "Tag-Prime",
    "llama3b512r": "Llama-3b",
    "llama8b256r": "Llama-8b",
    "mistral24b256r": "Mistral-24b",
}

##  Set up Figure ##

fig = Figure(
    figsize=(11, 7.5), layout="constrained"
)  # Use 'constrained' layout to optimize space
axes = fig.subplots(2, 3)

plot_answer_matrix(
    axes[0, 0],
    zero_argument_data.values,
    "Zero-shot\nArgument Extraction",
    zero_argument_data.columns.tolist(),
)
plot_answer_matrix(
    axes[0, 1],
    example_argument_data.values,
    "Few-shot\nArgument Extraction",
    example_argument_data.columns.tolist(),
)
plot_answer_matrix(
    axes[0, 2],
    tuned_argument_data.values,
    "Fine-Tuned\nArgument Extraction",
    [tuned_map[label] for label in tuned_argument_data.columns],
)

plot_answer_matrix(
    axes[1, 0],
    zero_event_data.values,
    "Zero-shot\nEvent Detection",
    zero_event_data.columns.tolist(),
)
plot_answer_matrix(
    axes[1, 1],
    example_event_data.values,
    "Few-shot\nEvent Detection",
    example_event_data.columns.tolist(),
)
plot_answer_matrix(
    axes[1, 2],
    example_event_type_data.values,
    "Few-shot\nEvent Type Detection",
    example_event_type_data.columns.tolist(),
)

# Add blank labels for space for the suplabels
axes[0, 0].set_ylabel(" ")
axes[1, 1].set_xlabel(" ")

fig.supxlabel("Predicted answers", y=0.005)  # Adjusted y position to bring closer
fig.supylabel("Referenced answers", x=0.005)  # Adjusted x position to bring closer

# Save the figure
fig.savefig(this_file.parent / f"{this_file.stem}.pdf")
