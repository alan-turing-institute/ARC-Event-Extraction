"""
Figure 4: Fine-tuning for Argument Extraction
"""

from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Load data
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FINE TUNING EAE DATA ##

fine_tuning_eae_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure4/fine_tuning_eae.csv")
)

# Custom X-axis

x_axis_values = [0, 16, 32, 64, 128, 256, 512, 1024]
x_axis_labels = [
    "Base\nModel",
    "...",
    "32",
    "64",
    "128",
    "256",
    "512",
    "1024",
]

# Convert fractional values to percentage
fine_tuning_eae_data["argument_f1"] = fine_tuning_eae_data["argument_f1"] * 100
fine_tuning_eae_data["bs_std"] = fine_tuning_eae_data["bs_std"] * 100

# New x-axis values
fine_tuning_eae_data["x_axis"] = fine_tuning_eae_data["r_value"].map(
    {v: n for n, v in enumerate(x_axis_values)}
)

# Filter data for each model
mistral_24b = fine_tuning_eae_data[fine_tuning_eae_data["model"] == "Mistral-24b"]
llama_3b = fine_tuning_eae_data[fine_tuning_eae_data["model"] == "Llama-3b"]
llama_8b = fine_tuning_eae_data[fine_tuning_eae_data["model"] == "Llama-8b"]
llama_70b = fine_tuning_eae_data[fine_tuning_eae_data["model"] == "Llama-70b"]

## MISTRAL DATA ##

mistral_ft_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure4/fine_tuning_mistral.csv")
)
mistral_ft_data["argument_f1"] = mistral_ft_data["argument_f1"] * 100
mistral_ft_data["bs_std"] = mistral_ft_data["bs_std"] * 100
mistral_ft_data["x_axis"] = mistral_ft_data["r_value"].map(
    {v: n for n, v in enumerate(x_axis_values)}
)

mistral_eae1 = mistral_ft_data[mistral_ft_data["prompt"] == "EAE1"]
mistral_eae6 = mistral_ft_data[mistral_ft_data["prompt"] == "EAE6"]


##  Set up Figure ##

fig = Figure(figsize=(6, 5), layout="tight")  # Changed from (8, 5) to (6, 5)
ax1 = fig.subplots(1, 1)

## PANEL A ##

# Model color
model_colors = {
    "Mistral-24b": "red",
    "Llama-3b": "green",
    "Llama-8b": "orange",
    "Llama-70b": "teal",
}
model_markers = {
    "Mistral-24b": "D",
    "Llama-3b": "^",
    "Llama-8b": "v",
    "Llama-70b": "o",
}

for model, data in [
    ("Mistral-24b", mistral_24b),
    ("Llama-70b", llama_70b),
    ("Llama-8b", llama_8b),
    ("Llama-3b", llama_3b),
]:
    # Plot the data with error bars
    ax1.errorbar(
        data["x_axis"],
        data["argument_f1"],
        yerr=data["bs_std"],
        fmt=model_markers[model],
        label=model,
        color=model_colors[model],
        capsize=8,
    )
    ax1.plot(
        data["x_axis"].iloc[:2],
        data["argument_f1"].iloc[:2],
        linestyle=":",
        color=model_colors[model],
    )
    ax1.plot(
        data["x_axis"].iloc[1:],
        data["argument_f1"].iloc[1:],
        linestyle="-",
        color=model_colors[model],
    )


ax1.text(
    0.02,
    0.98,
    "Fine-Tuning Argument Extraction",
    transform=ax1.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax1.set_ylabel("Argument F1")
ax1.set_xlabel("Rank of PEFT adapter", labelpad=-5)  # Moved up because two line tick
ax1.set_ylim(25, 85)

ax1.set_xticks(range(len(x_axis_values)))
ax1.set_xticklabels(x_axis_labels)

ax1.grid(axis="y")
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(axis="y", direction="in", which="both", left=True, right=True)
ax1.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.grid(axis="x")

# Update legend to include new data
ax1.legend(bbox_to_anchor=(0.01, 0.95), loc="upper left")

# INSET AXIS ##
inset_ax = ax1.inset_axes([0.55, 0.07, 0.42, 0.40])  # [x, y, width, height]

# Plot data on inset axis

inset_ax.errorbar(
    mistral_eae6["x_axis"],
    mistral_eae6["argument_f1"],
    yerr=mistral_eae6["bs_std"],
    fmt="+",
    ls="-",
    lw=1,
    label="Few-shot",
    color="#990f4b",
    capsize=4,
)
inset_ax.errorbar(
    mistral_eae1["x_axis"],
    mistral_eae1["argument_f1"],
    yerr=mistral_eae1["bs_std"],
    fmt="x",
    ls="--",
    lw=1,
    label="Zero-shot",
    color="blue",
    capsize=4,
)

# Customize inset axis

x_axis_labels[0] = "Base"
inset_ax.set_ylim(20, 85)
inset_ax.set_xticks(range(len(x_axis_values[:6])))
inset_ax.set_xticklabels(x_axis_labels[:6])
inset_ax.grid(axis="both", linestyle="--", alpha=0.5)

inset_ax.legend(loc="lower right", fontsize=9, title="Mistral-24b", title_fontsize=9)

# Save the figure
fig.savefig(os.path.join(main_dir, "report-figures/figure4/figure4.pdf"))
