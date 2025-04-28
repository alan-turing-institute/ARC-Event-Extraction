from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Load data
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

# Load EAE data
eae_data = pd.read_csv(
    os.path.join(
        main_dir, "report-figures/sup-figures/other_in_context/other_prompt_eae.csv"
    )
)

# Convert Fraction to Percentage
eae_data["argument_f1"] = eae_data["argument_f1"] * 100
eae_data["bs_std"] = eae_data["bs_std"] * 100

# Filter data for in-context prompts
eae_none = eae_data[eae_data["prompt"] == "EAE1"]
eae_random = eae_data[eae_data["prompt"] == "EAE3"]
eae_close = eae_data[eae_data["prompt"] == "EAE6"]
eae_mask = eae_data[eae_data["prompt"] == "EAE8"]

##  Set up Figure ##

fig = Figure(figsize=(7, 7), layout="tight")
axes = fig.subplots(2, 2)


def plot_axis(axis, model_choice):
    this_eae_none = eae_none[eae_none["model"] == model_choice]
    this_eae_random = eae_random[eae_random["model"] == model_choice]
    this_eae_close = eae_close[eae_close["model"] == model_choice]
    this_eae_mask = eae_mask[eae_mask["model"] == model_choice]

    axis.errorbar(
        this_eae_mask["examples"],
        this_eae_mask["argument_f1"],
        yerr=this_eae_mask["bs_std"],
        fmt="^",
        label="Masked event",
        color="purple",
        capsize=8,
    )
    axis.errorbar(
        this_eae_close["examples"],
        this_eae_close["argument_f1"],
        yerr=this_eae_close["bs_std"],
        fmt="v",
        label="Close same event",
        color="orange",
        capsize=8,
    )
    axis.errorbar(
        this_eae_random["examples"],
        this_eae_random["argument_f1"],
        yerr=this_eae_random["bs_std"],
        fmt="o",
        label="Random any event",
        color="blue",
        capsize=8,
    )
    axis.errorbar(
        this_eae_none["examples"],
        this_eae_none["argument_f1"],
        yerr=this_eae_none["bs_std"],
        fmt="s",
        label="No examples",
        color="red",
        capsize=8,
    )

    # Add lines connecting "No examples" to other EAE datasets
    axis.plot(
        this_eae_none["examples"].tolist() + this_eae_random["examples"].tolist(),
        this_eae_none["argument_f1"].tolist() + this_eae_random["argument_f1"].tolist(),
        color="blue",
        linestyle=":",
        linewidth=1.2,
        label=None,
    )
    axis.plot(
        this_eae_none["examples"].tolist() + this_eae_close["examples"].tolist(),
        this_eae_none["argument_f1"].tolist() + this_eae_close["argument_f1"].tolist(),
        color="orange",
        linestyle=":",
        linewidth=1.2,
        label=None,
    )
    axis.plot(
        this_eae_none["examples"].tolist() + this_eae_mask["examples"].tolist(),
        this_eae_none["argument_f1"].tolist() + this_eae_mask["argument_f1"].tolist(),
        color="purple",
        linestyle=":",
        linewidth=1.2,
        label=None,
    )

    axis.text(
        0.02,
        0.98,
        f"{model_choice}",
        transform=axis.transAxes,
        fontsize=12,
        va="top",
        ha="left",
    )

    axis.set_ylabel("Argument F1")
    axis.set_xlabel("Number of examples")
    axis.set_ylim(20, 62)
    axis.set_xlim(-2, 22)
    axis.grid(axis="y")
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))
    axis.tick_params(axis="y", direction="in", which="both", left=True, right=True)
    axis.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.grid(axis="x")


# Plot The Models

plot_axis(axes[0, 0], "Phi-14b")
plot_axis(axes[0, 1], "Gemma-27b")
plot_axis(axes[1, 0], "Llama-70b")
plot_axis(axes[1, 1], "Mistral-123b")

axes[1, 1].legend(loc="lower right")

# Save the figure
fig.savefig(
    os.path.join(
        main_dir, "report-figures/sup-figures/other_in_context/other_eae_in_context.pdf"
    )
)
