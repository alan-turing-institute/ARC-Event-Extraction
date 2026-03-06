"""
Figure: Comparison of event extraction performance across different models and
evaluation regimes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from arcee.plotting import plot_answer_matrix
from matplotlib.figure import Figure

# Load data
this_file = Path(__file__).resolve()

zero_event_data = pd.read_csv(this_file.parent / "zero_event_f1.csv")
example_event_data = pd.read_csv(this_file.parent / "example_event_f1.csv")
example_event_type_data = pd.read_csv(this_file.parent / "example_event_type_f1.csv")

##  Set up Figure ##

fig = Figure(
    figsize=(4, 10.5), layout="constrained"
)  # Use 'constrained' layout to optimize space
axes = fig.subplots(3, 1)

plot_answer_matrix(
    axes[0],
    zero_event_data.values,
    "Zero-shot Event Detection",
    zero_event_data.columns.tolist(),
)
plot_answer_matrix(
    axes[1],
    example_event_data.values,
    "Few-shot Event Detection",
    example_event_data.columns.tolist(),
)
plot_answer_matrix(
    axes[2],
    example_event_type_data.values,
    "Few-shot Event Type Detection",
    example_event_type_data.columns.tolist(),
)

# Add blank labels for space for the suplabels
for ax in axes:
    ax.set_ylabel(" ")
axes[-1].set_xlabel(" ")

fig.supxlabel("Predicted answers", y=0.005)  # Adjusted y position to bring closer
fig.supylabel("Referenced answers", x=0.005)  # Adjusted x position to bring closer

# Save the figure
fig.savefig(this_file.parent / f"{this_file.stem}.pdf")
