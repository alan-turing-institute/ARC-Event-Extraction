from __future__ import annotations

import os

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Load data
main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

# Load ED data
ed_data = pd.read_csv(
    os.path.join(
        main_dir, "report-figures/sup-figures/other_in_context/other_prompt_ed.csv"
    )
)

# Convert Fraction to Percentage
ed_data["event_f1"] = ed_data["event_f1"] * 100
ed_data["bs_std"] = ed_data["bs_std"] * 100

# Filter data for in-context prompts
ed_none = ed_data[ed_data["prompt"] == "ED1"]
ed_random = ed_data[ed_data["prompt"] == "ED3"]
ed_close = ed_data[ed_data["prompt"] == "ED4"]
ed_mask = ed_data[ed_data["prompt"] == "ED8"]

##  Set up Figure ##

fig = Figure(figsize=(7, 7), layout="tight")
axes = fig.subplots(2, 2)


def plot_axis(axis, model_choice):
    this_ed_none = ed_none[ed_none["model"] == model_choice]
    this_ed_random = ed_random[ed_random["model"] == model_choice]
    this_ed_close = ed_close[ed_close["model"] == model_choice]
    this_ed_mask = ed_mask[ed_mask["model"] == model_choice]

    axis.errorbar(
        this_ed_mask["examples"],
        this_ed_mask["event_f1"],
        yerr=this_ed_mask["bs_std"],
        fmt="^",
        label="Masked event",
        color="purple",
        capsize=8,
    )
    axis.errorbar(
        this_ed_close["examples"],
        this_ed_close["event_f1"],
        yerr=this_ed_close["bs_std"],
        fmt="v",
        label="Close same event",
        color="orange",
        capsize=8,
    )
    axis.errorbar(
        this_ed_random["examples"],
        this_ed_random["event_f1"],
        yerr=this_ed_random["bs_std"],
        fmt="o",
        label="Random any event",
        color="blue",
        capsize=8,
    )
    axis.errorbar(
        this_ed_none["examples"],
        this_ed_none["event_f1"],
        yerr=this_ed_none["bs_std"],
        fmt="s",
        label="No examples",
        color="red",
        capsize=8,
    )

    # Add lines connecting "No examples" to other ED datasets
    axis.plot(
        this_ed_none["examples"].tolist() + this_ed_random["examples"].tolist(),
        this_ed_none["event_f1"].tolist() + this_ed_random["event_f1"].tolist(),
        color="blue",
        linestyle=":",
        linewidth=1.2,
        label=None,
    )
    axis.plot(
        this_ed_none["examples"].tolist() + this_ed_close["examples"].tolist(),
        this_ed_none["event_f1"].tolist() + this_ed_close["event_f1"].tolist(),
        color="orange",
        linestyle=":",
        linewidth=1.2,
        label=None,
    )
    axis.plot(
        this_ed_none["examples"].tolist() + this_ed_mask["examples"].tolist(),
        this_ed_none["event_f1"].tolist() + this_ed_mask["event_f1"].tolist(),
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

    axis.set_ylabel("Event F1")
    axis.set_xlabel("Number of examples")
    axis.set_ylim(0, 60)
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
        main_dir, "report-figures/sup-figures/other_in_context/other_ed_in_context.pdf"
    )
)
