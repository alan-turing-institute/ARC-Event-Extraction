"""
Appendix Figure: Event type performance EE models
"""

from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Load data
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

ed_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/event_type_performance/ed_model_performance.csv",
    )
)
ed_type_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/event_type_performance/ed_model_type_performance.csv",
    )
)

model_list = [
    "Llama-8b",
    "Phi-14b",
    "Mistral-24b",
    "Gemma-27b",
    "Llama-70b",
    "Mistral-123b",
    "GPT-4o",
]

# Convert Fraction to Percentage
ed_data["event_f1"] = ed_data["event_f1"] * 100
ed_data["bs_std"] = ed_data["bs_std"] * 100
ed_type_data["event_type_f1"] = ed_type_data["event_type_f1"] * 100
ed_type_data["bs_std"] = ed_type_data["bs_std"] * 100

# Filter data for in context prompts
ed_in_context = ed_data[ed_data["prompt"] == "ED4"]
ed_type_in_context = ed_type_data[ed_type_data["prompt"] == "ED4"]

# Filter zero-shot data
ed_zero_shot = ed_data[(ed_data["prompt"] == "ED1")]
ed_type_zero_shot = ed_type_data[(ed_type_data["prompt"] == "ED1")]

##  Set up Figure ##

fig = Figure(figsize=(8, 4.5), layout="tight")
axes = fig.subplots(1, 2)

ax1 = axes[0]
ax2 = axes[1]

## PANEL A ##

ed_in_context = ed_in_context.set_index("model").reindex(model_list).reset_index()
ax1.errorbar(
    model_list,
    ed_in_context["event_f1"],
    yerr=ed_in_context["bs_std"],
    fmt="o",
    label="Few Shot",
    color="blue",
    capsize=8,
)
ed_zero_shot = ed_zero_shot.set_index("model").reindex(model_list).reset_index()
ax1.errorbar(
    model_list,
    ed_zero_shot["event_f1"],
    yerr=ed_zero_shot["bs_std"],
    fmt="s",
    label="Zero Shot",
    color="red",
    capsize=8,
)
ax1.axhline(
    y=71.3,
    color="green",
    linestyle="--",
    linewidth=2,
    label="TextEE Best",
)
ax1.axhline(
    y=14.6,
    color="#a3b012",
    linestyle="--",
    linewidth=2,
    label="TextEE LLM Best",
)
ax1.text(
    0.01,
    0.99,
    "Event Detection\nType and Trigger Matching",
    transform=ax1.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax1.set_ylabel("Event F1")
ax1.set_ylim(0, 85)
ax1.set_xlim(-0.5, len(model_list) - 0.5)
ax1.set_xticks(model_list)
ax1.set_xticklabels(model_list, rotation=45, ha="right")
ax1.grid(axis="y")
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(axis="y", direction="in", which="both", left=True, right=True)

## PANEL B ##

ed_type_in_context = (
    ed_type_in_context.set_index("model").reindex(model_list).reset_index()
)
ax2.errorbar(
    model_list,
    ed_type_in_context["event_type_f1"],
    yerr=ed_type_in_context["bs_std"],
    fmt="o",
    label="In Context",
    color="blue",
    capsize=8,
)
ed_type_zero_shot = (
    ed_type_zero_shot.set_index("model").reindex(model_list).reset_index()
)
ax2.errorbar(
    model_list,
    ed_type_zero_shot["event_type_f1"],
    yerr=ed_type_zero_shot["bs_std"],
    fmt="s",
    label="Zero Shot",
    color="red",
    capsize=8,
)
ax2.axhline(
    y=71.3,
    color="green",
    linestyle="--",
    linewidth=2,
    label="TextEE Best",
)
ax2.axhline(
    y=14.6,
    color="#a3b012",
    linestyle="--",
    linewidth=2,
    label="TextEE LLM Best",
)
ax2.text(
    0.01,
    0.99,
    "Event Detection\nOnly Type Matching",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax2.set_ylabel("Event Type F1")
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(model_list)
ax2.set_xticklabels(model_list, rotation=45, ha="right")
ax2.grid(axis="y")
ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(axis="y", direction="in", which="both", left=True, right=True)

# Add Legend #
handles, labels = ax2.get_legend_handles_labels()
leg_order = [2, 3, 0, 1]
ax2.legend(
    [handles[i] for i in leg_order],
    [labels[i] for i in leg_order],
    loc="upper right",
    ncol=1,
    bbox_to_anchor=(1.0, 0.48),
    bbox_transform=ax2.transAxes,
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

# Save the figure
fig.savefig(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/event_type_performance/event_type_performance.pdf",
    )
)
