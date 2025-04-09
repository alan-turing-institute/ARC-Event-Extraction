"""
Figure 2: Headline Results
"""

from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Load data
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

eae_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure2/eae_model_performance.csv")
)
ed_data = pd.read_csv(
    os.path.join(main_dir, "report-figures/figure2/ed_model_performance.csv")
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
eae_data["argument_f1"] = eae_data["argument_f1"] * 100
eae_data["bs_std"] = eae_data["bs_std"] * 100
ed_data["event_f1"] = ed_data["event_f1"] * 100
ed_data["bs_std"] = ed_data["bs_std"] * 100

# Filter data for in context prompts
eae_in_context = eae_data[eae_data["prompt"] == "EAE6"]
ed_in_context = ed_data[ed_data["prompt"] == "ED8"]

# Filter zero-shot data
eae_zero_shot = eae_data[(eae_data["prompt"] == "EAE1") & (~eae_data["fine_tune"])]
ed_zero_shot = ed_data[(ed_data["prompt"] == "ED1")]

# Filter fine-tuned EAE1 data
eae_fine_tuned = eae_data[(eae_data["prompt"] == "EAE1") & (eae_data["fine_tune"])]

##  Set up Figure ##

fig = Figure(figsize=(8, 5), layout="tight")
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
    label="In Context",
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
ax1.text(
    0.01,
    0.99,
    "Event Detection",
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
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(axis="y", direction="in", which="both", left=True, right=True)


## PANEL B ##

eae_fine_tuned = eae_fine_tuned.set_index("model").reindex(model_list).reset_index()
ax2.errorbar(
    model_list,
    eae_fine_tuned["argument_f1"],
    yerr=eae_fine_tuned["bs_std"],
    fmt="D",
    label="Fine-tuned",
    color="magenta",
    capsize=8,
)
eae_in_context = eae_in_context.set_index("model").reindex(model_list).reset_index()
ax2.errorbar(
    model_list,
    eae_in_context["argument_f1"],
    yerr=eae_in_context["bs_std"],
    fmt="o",
    label="Few shot",
    color="blue",
    capsize=8,
)
eae_zero_shot = eae_zero_shot.set_index("model").reindex(model_list).reset_index()
ax2.errorbar(
    model_list,
    eae_zero_shot["argument_f1"],
    yerr=eae_zero_shot["bs_std"],
    fmt="s",
    label="Zero shot",
    color="red",
    capsize=8,
)
ax2.text(
    0.01,
    0.99,
    "Event Argumentation Extraction",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax2.set_ylabel("Argument F1")
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(model_list)
ax2.set_xticklabels(model_list, rotation=45, ha="right")
ax2.grid(axis="y")
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(axis="y", direction="in", which="both", left=True, right=True)


# Add Legend #

handles, labels = ax2.get_legend_handles_labels()
ax1.legend(
    handles,
    labels,
    loc="upper right",
    title="Prompt Type",
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
fig.savefig(os.path.join(main_dir, "report-figures/figure2/figure2.pdf"))
