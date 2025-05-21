from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Load data
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

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

# Calculate the difference from missing values
example_event_diff = example_event_recall.values - example_event_precision.values.T
example_argument_diff = (
    example_argument_recall.values - example_argument_precision.values.T
)
zero_argument_diff = zero_argument_recall.values - zero_argument_precision.values.T
tuned_argument_diff = tuned_argument_recall.values - tuned_argument_precision.values.T

# Print the maximum difference
print("Example Event Diff")
print(np.max(np.abs(example_event_diff)))
print("Example Argument Diff")
print(np.max(np.abs(example_argument_diff)))
print("Zero Argument Diff")
print(np.max(np.abs(zero_argument_diff)))
print("Tuned Argument Diff")
print(np.max(np.abs(tuned_argument_diff)))
