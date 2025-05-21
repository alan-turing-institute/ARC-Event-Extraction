"""
RAMS Performance: Panel A (Model Comparison) and Panel B (Effect of Examples)
"""

from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Set top-level directory
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

# Load data for Panel A
model_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/rams_performance/rams_eae_model_performance.csv",
    )
)

# Load data for Panel B
example_data = pd.read_csv(
    os.path.join(
        main_dir,
        "report-figures/sup-figures/rams_performance/rams_mistral24b_prompt_eae.csv",
    )
)

# Convert Fraction to Percentage
model_data["argument_f1"] = model_data["argument_f1"] * 100
model_data["bs_std"] = model_data["bs_std"] * 100
example_data["argument_f1"] = example_data["argument_f1"] * 100
example_data["bs_std"] = example_data["bs_std"] * 100

# Filter data for Panel A
model_list = ["Llama-8b", "Phi-14b", "Mistral-24b", "Llama-70b", "Mistral-123b"]
zero_shot = model_data[model_data["prompt"] == "EAE1"]
rand_shot = model_data[model_data["prompt"] == "EAE3"]
few_shot = model_data[model_data["prompt"] == "EAE6"]

# Filter data for Panel B
eae_none = example_data[example_data["prompt"] == "EAE1"]
eae_random = example_data[example_data["prompt"] == "EAE3"]
eae_similar = example_data[example_data["prompt"] == "EAE6"]

## Set up Figure ##

fig = Figure(figsize=(8, 5), layout="tight")
axes = fig.subplots(1, 2)

ax1 = axes[0]
ax2 = axes[1]

## PANEL A ##

few_shot = few_shot.set_index("model").reindex(model_list).reset_index()
ax1.errorbar(
    model_list,
    few_shot["argument_f1"],
    yerr=few_shot["bs_std"],
    fmt="D",
    label="Similar examples",
    color="#c947ba",
    capsize=8,
)
rand_shot = rand_shot.set_index("model").reindex(model_list).reset_index()
ax1.errorbar(
    model_list,
    rand_shot["argument_f1"],
    yerr=rand_shot["bs_std"],
    fmt="o",
    label="Random examples",
    color="blue",
    capsize=8,
)
zero_shot = zero_shot.set_index("model").reindex(model_list).reset_index()
ax1.errorbar(
    model_list,
    zero_shot["argument_f1"],
    yerr=zero_shot["bs_std"],
    fmt="s",
    label="Zero shot",
    color="red",
    capsize=8,
)
ax1.axhline(
    y=50.5,
    color="green",
    linestyle="--",
    linewidth=2,
    label="TextEE Best",
)
ax1.text(
    0.03,
    0.98,
    "Event Argument Extraction",
    transform=ax1.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)
ax1.set_ylabel("Argument F1")
ax1.set_ylim(0, 55)
ax1.set_xlim(-0.5, len(model_list) - 0.5)
ax1.set_xticks(model_list)
ax1.set_xticklabels(model_list, rotation=45, ha="right")
ax1.grid(axis="y")
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(axis="y", direction="in", which="both", left=True, right=True)
ax1.legend(loc="lower right")

## PANEL B ##

ax2.errorbar(
    eae_similar["examples"],
    eae_similar["argument_f1"],
    yerr=eae_similar["bs_std"],
    fmt="D",
    label="Similar examples",
    color="#c947ba",
    capsize=8,
)
ax2.errorbar(
    eae_random["examples"],
    eae_random["argument_f1"],
    yerr=eae_random["bs_std"],
    fmt="o",
    label="Random examples",
    color="blue",
    capsize=8,
)
ax2.errorbar(
    eae_none["examples"],
    eae_none["argument_f1"],
    yerr=eae_none["bs_std"],
    fmt="s",
    label="No examples",
    color="red",
    capsize=8,
)
ax2.plot(
    eae_none["examples"].tolist() + eae_random["examples"].tolist(),
    eae_none["argument_f1"].tolist() + eae_random["argument_f1"].tolist(),
    color="blue",
    linestyle=":",
    linewidth=1.2,
    label=None,
)
ax2.plot(
    eae_none["examples"].tolist() + eae_similar["examples"].tolist(),
    eae_none["argument_f1"].tolist() + eae_similar["argument_f1"].tolist(),
    color="#c947ba",
    linestyle=":",
    linewidth=1.2,
    label=None,
)
ax2.text(
    0.03,
    0.97,
    "Effect of Examples\nMistral 24B",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)
ax2.text(
    0.03,
    0.03,
    "Roles Across Multiple Sentences\n(RAMS)",
    transform=ax2.transAxes,
    fontsize=12,
    va="bottom",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)
ax2.set_ylabel("Argument F1")
ax2.set_xlabel("Number of examples")
ax2.set_ylim(ax1.get_ylim())
ax2.grid(axis="y")
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(axis="y", direction="in", which="both", left=True, right=True)
ax2.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.grid(axis="x")

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
        main_dir, "report-figures/sup-figures/rams_performance/rams_performance.pdf"
    )
)
