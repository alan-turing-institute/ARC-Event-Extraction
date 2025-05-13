# Re-import necessary libraries after code execution state reset
from __future__ import annotations

import os

import colorcet as cc
import pandas as pd
from matplotlib.figure import Figure

# Recreate the dataset since state was reset

main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

ac_df = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/textee_datasets/textee_ac.csv",
    ),
    index_col=0,
    header=0,
)

tc_df = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/textee_datasets/textee_tc.csv",
    ),
    index_col=0,
    header=0,
)

# Set up the figure

fig = Figure(figsize=(8, 9))
axes = fig.subplots(2, 1)
ax1 = axes[0]
ax2 = axes[1]


# Convert to long format for plotting


def plot_model_scores(data_df, ax):
    """
    Plot model scores on the given axis.

    Args:
        data_df (pd.DataFrame): DataFrame containing model scores.
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    d_long_df = data_df.reset_index().melt(
        id_vars="index", var_name="Dataset", value_name="Score"
    )
    d_long_df.rename(columns={"index": "Model"}, inplace=True)
    d_long_df = d_long_df.dropna()

    # Calculate average success rate for each model and sort models by it
    model_avg_scores = (
        d_long_df.groupby("Model")["Score"].mean().sort_values(ascending=False)
    )
    ordered_models = model_avg_scores.index.tolist()

    color_div = len(ordered_models) / len(cc.rainbow4)
    color_map = {
        model: cc.rainbow4[round(i / color_div)]
        for i, model in enumerate(ordered_models)
    }

    for model in ordered_models:
        model_data = d_long_df[d_long_df["Model"] == model]
        ax.scatter(
            model_data["Dataset"],
            model_data["Score"],
            label=model,
            alpha=0.3,
            s=200,
            # edgecolors='black',
            linewidths=1,
            color=color_map[model],  # Assign color from the colormap
        )

    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 100)

    # Configure y-axis ticks
    ax.tick_params(axis="y", which="both", direction="in", right=True, left=True)
    ax.yaxis.minorticks_on()

    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)

    ax.grid()


# Plot the data
plot_model_scores(tc_df, ax1)
ax1.set_ylabel("Event F1")

plot_model_scores(ac_df, ax2)
ax2.set_ylabel("Argument F1")

# Add tiles
ax1.text(
    0.03,
    0.97,
    "TextEE Event Detection",
    transform=ax1.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)
ax2.text(
    0.03,
    0.97,
    "TextEE Event Argument Extraction",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)
# Add subplot labels
ax1.text(
    -0.08,
    1.0,
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

fig.tight_layout()
fig.savefig(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/textee_datasets/textee_datasets.pdf",
    ),
)
