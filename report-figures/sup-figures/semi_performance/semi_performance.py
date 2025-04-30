"""
Appendix Figure: Semi-string performance EE models
"""

from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Load data
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

ed_data = pd.read_csv(
    os.path.join(
        main_dir, "report-figures/sup-figures/semi_performance/ed_model_performance.csv"
    )
)
ed_semi_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/semi_performance/ed_model_semi_performance.csv",
    )
)
eae_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/semi_performance/eae_model_performance.csv",
    )
)
eae_semi_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/semi_performance/eae_model_semi_performance.csv",
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
ed_semi_data["event_f1"] = ed_semi_data["event_f1"] * 100
ed_semi_data["bs_std"] = ed_semi_data["bs_std"] * 100
eae_data["argument_f1"] = eae_data["argument_f1"] * 100
eae_data["bs_std"] = eae_data["bs_std"] * 100
eae_semi_data["argument_f1"] = eae_semi_data["argument_f1"] * 100
eae_semi_data["bs_std"] = eae_semi_data["bs_std"] * 100

# Filter data for in context prompts
ed_in_context = ed_data[ed_data["prompt"] == "ED4"]
ed_semi_in_context = ed_semi_data[ed_semi_data["prompt"] == "ED4"]
eae_in_context = eae_data[eae_data["prompt"] == "EAE6"]
eae_semi_in_context = eae_semi_data[eae_semi_data["prompt"] == "EAE6"]

# Filter zero-shot data
ed_zero_shot = ed_data[(ed_data["prompt"] == "ED1")]
ed_semi_zero_shot = ed_semi_data[(ed_semi_data["prompt"] == "ED1")]
eae_zero_shot = eae_data[(eae_data["prompt"] == "EAE1") & (~eae_data["fine_tune"])]
eae_semi_zero_shot = eae_semi_data[
    (eae_semi_data["prompt"] == "EAE1") & (~eae_semi_data["fine_tune"])
]

# Filter fine-tuned EAE1 data
eae_fine_tuned = eae_data[(eae_data["prompt"] == "EAE1") & (eae_data["fine_tune"])]
eae_semi_fine_tuned = eae_semi_data[
    (eae_semi_data["prompt"] == "EAE1") & (eae_semi_data["fine_tune"])
]
##  Set up Figure ##

fig = Figure(figsize=(8, 8), layout="tight")
axes = fig.subplots(2, 2)

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

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

ed_semi_in_context = (
    ed_semi_in_context.set_index("model").reindex(model_list).reset_index()
)
ax2.errorbar(
    model_list,
    ed_semi_in_context["event_f1"],
    yerr=ed_semi_in_context["bs_std"],
    fmt="o",
    label="In Context",
    color="blue",
    capsize=8,
)
ed_semi_zero_shot = (
    ed_semi_zero_shot.set_index("model").reindex(model_list).reset_index()
)
ax2.errorbar(
    model_list,
    ed_semi_zero_shot["event_f1"],
    yerr=ed_semi_zero_shot["bs_std"],
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
    "Event Detection\nPartial String Matching",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax2.set_ylabel("Event F1")
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(model_list)
ax2.set_xticklabels(model_list, rotation=45, ha="right")
ax2.grid(axis="y")
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(axis="y", direction="in", which="both", left=True, right=True)

## PANEL C ##

eae_fine_tuned = eae_fine_tuned.set_index("model").reindex(model_list).reset_index()
ax3.errorbar(
    model_list,
    eae_fine_tuned["argument_f1"],
    yerr=eae_fine_tuned["bs_std"],
    fmt="D",
    label="Fine-tuned",
    color="magenta",
    capsize=8,
)
eae_in_context = eae_in_context.set_index("model").reindex(model_list).reset_index()
ax3.errorbar(
    model_list,
    eae_in_context["argument_f1"],
    yerr=eae_in_context["bs_std"],
    fmt="o",
    label="Few shot",
    color="blue",
    capsize=8,
)
eae_zero_shot = eae_zero_shot.set_index("model").reindex(model_list).reset_index()
ax3.errorbar(
    model_list,
    eae_zero_shot["argument_f1"],
    yerr=eae_zero_shot["bs_std"],
    fmt="s",
    label="Zero shot",
    color="red",
    capsize=8,
)
ax3.axhline(
    y=77.8,
    color="green",
    linestyle="--",
    linewidth=2,
    label="TextEE Best",
)
ax3.axhline(
    y=29.2,
    color="#a3b012",
    linestyle="--",
    linewidth=2,
    label="TextEE LLM Best",
)
ax3.text(
    0.01,
    0.99,
    "Event Argumentation Extraction",
    transform=ax3.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax3.plot([0, 0], [0, 0], "o", color="White", label=" ")  # Dummy plot for legend

ax3.set_ylabel("Argument F1")
ax3.set_ylim(0, 95)
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(model_list)
ax3.set_xticklabels(model_list, rotation=45, ha="right")
ax3.grid(axis="y")
ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
ax3.tick_params(axis="y", direction="in", which="both", left=True, right=True)

## PANEL D ##

eae_semi_fine_tuned = (
    eae_semi_fine_tuned.set_index("model").reindex(model_list).reset_index()
)
ax4.errorbar(
    model_list,
    eae_semi_fine_tuned["argument_f1"],
    yerr=eae_semi_fine_tuned["bs_std"],
    fmt="D",
    label="Fine-tuned",
    color="magenta",
    capsize=8,
)
eae_semi_in_context = (
    eae_semi_in_context.set_index("model").reindex(model_list).reset_index()
)
ax4.errorbar(
    model_list,
    eae_semi_in_context["argument_f1"],
    yerr=eae_semi_in_context["bs_std"],
    fmt="o",
    label="Few shot",
    color="blue",
    capsize=8,
)

eae_semi_zero_shot = (
    eae_semi_zero_shot.set_index("model").reindex(model_list).reset_index()
)
ax4.errorbar(
    model_list,
    eae_semi_zero_shot["argument_f1"],
    yerr=eae_semi_zero_shot["bs_std"],
    fmt="s",
    label="Zero shot",
    color="red",
    capsize=8,
)
ax4.axhline(
    y=77.8,
    color="green",
    linestyle="--",
    linewidth=2,
    label="TextEE Best",
)
ax4.axhline(
    y=29.2,
    color="#a3b012",
    linestyle="--",
    linewidth=2,
    label="TextEE LLM Best",
)
ax4.plot([0, 0], [0, 0], "o", color="White", label=" ")  # Dummy plot for legend
ax4.text(
    0.01,
    0.99,
    "Event Argumentation Extraction\nPartial String Matching",
    transform=ax4.transAxes,
    fontsize=12,
    va="top",
    ha="left",
)
ax4.set_ylabel("Argument F1")
ax4.set_ylim(ax3.get_ylim())
ax4.set_xlim(ax1.get_xlim())
ax4.set_xticks(model_list)
ax4.set_xticklabels(model_list, rotation=45, ha="right")
ax4.grid(axis="y")
ax4.yaxis.set_minor_locator(AutoMinorLocator(5))
ax4.tick_params(axis="y", direction="in", which="both", left=True, right=True)

# Add Legend #
handles, labels = ax4.get_legend_handles_labels()
leg_order = [3, 4, 5, 2, 0, 1]
ax4.legend(
    [handles[i] for i in leg_order],
    [labels[i] for i in leg_order],
    loc="lower left",
    ncol=2,
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
ax3.text(
    -0.08,
    1.0,
    "C",
    transform=ax3.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax4.text(
    -0.08,
    1.0,
    "D",
    transform=ax4.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)

# Save the figure
fig.savefig(
    os.path.join(
        main_dir, "report-figures/sup-figures/semi_performance/semi_performance.pdf"
    )
)
