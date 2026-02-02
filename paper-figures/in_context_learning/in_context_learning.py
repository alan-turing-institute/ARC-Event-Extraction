"""
Figure 3: Effect of In-context Examples
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Load data
this_file = Path(__file__).resolve()

ed_data = pd.read_csv(this_file.parent / "mistral24b_prompt_ed.csv")

# Convert Fraction to Percentage

ed_data["event_f1"] = ed_data["event_f1"] * 100
ed_data["bs_std"] = ed_data["bs_std"] * 100

# Filter data for in context prompts
ed_none = ed_data[ed_data["prompt"] == "ED1"]
ed_random = ed_data[ed_data["prompt"] == "ED3"]
ed_close = ed_data[ed_data["prompt"] == "ED4"]

# Load EAE data
eae_data = pd.read_csv(this_file.parent / "mistral24b_prompt_eae.csv")

# Convert Fraction to Percentage
eae_data["argument_f1"] = eae_data["argument_f1"] * 100
eae_data["bs_std"] = eae_data["bs_std"] * 100

# Filter data for in-context prompts
eae_none = eae_data[eae_data["prompt"] == "EAE1"]
eae_random_any = eae_data[eae_data["prompt"] == "EAE3"]
eae_random_same = eae_data[eae_data["prompt"] == "EAE4"]
eae_close_any = eae_data[eae_data["prompt"] == "EAE5"]
eae_close_same = eae_data[eae_data["prompt"] == "EAE6"]

##  Set up Figure ##

fig = Figure(figsize=(6, 4.5), layout="tight")
axes = fig.subplots(1, 2)

ax1 = axes[0]
ax2 = axes[1]

## PANEL A ##

ax1.errorbar(
    ed_close["examples"],
    ed_close["event_f1"],
    yerr=ed_close["bs_std"],
    fmt="D",
    label="Similar examples",
    color="#c947ba",
    capsize=8,
)
ax1.errorbar(
    ed_random["examples"],
    ed_random["event_f1"],
    yerr=ed_random["bs_std"],
    fmt="o",
    label="Random examples",
    color="blue",
    capsize=8,
)
ax1.errorbar(
    ed_none["examples"],
    ed_none["event_f1"],
    yerr=ed_none["bs_std"],
    fmt="s",
    label="No examples",
    color="red",
    capsize=8,
)

ax1.plot(
    ed_none["examples"].tolist() + ed_random["examples"].tolist(),
    ed_none["event_f1"].tolist() + ed_random["event_f1"].tolist(),
    color="blue",
    linestyle=":",
    linewidth=1.2,
    label=None,
)
ax1.plot(
    ed_none["examples"].tolist() + ed_close["examples"].tolist(),
    ed_none["event_f1"].tolist() + ed_close["event_f1"].tolist(),
    color="#c947ba",
    linestyle=":",
    linewidth=1.2,
    label=None,
)

ax1.text(
    0.03,
    0.97,
    "Detection\nMistral 24B",
    transform=ax1.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)
ax1.set_ylabel("Event F1", fontsize=12)
ax1.set_xlabel("Number of examples", fontsize=12)
ax1.set_ylim(15, 62)

ax1.grid(axis="y")
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(axis="y", direction="in", which="both", left=True, right=True)
ax1.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.grid(axis="x")

ax1.legend(loc="lower right")


## PANEL B ##

ax2.errorbar(
    eae_close_same["examples"],
    eae_close_same["argument_f1"],
    yerr=eae_close_same["bs_std"],
    fmt="v",
    label="Similar same event",
    color="orange",
    capsize=8,
)
ax2.errorbar(
    eae_close_any["examples"],
    eae_close_any["argument_f1"],
    yerr=eae_close_any["bs_std"],
    fmt="D",
    label="Similar any event",
    color="#c947ba",
    capsize=8,
)
ax2.errorbar(
    eae_random_same["examples"],
    eae_random_same["argument_f1"],
    yerr=eae_random_same["bs_std"],
    fmt="^",
    label="Random same event",
    color="green",
    capsize=8,
)
ax2.errorbar(
    eae_random_any["examples"],
    eae_random_any["argument_f1"],
    yerr=eae_random_any["bs_std"],
    fmt="o",
    label="Random any event",
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

# Add lines connecting "No examples" to other EAE datasets
ax2.plot(
    eae_none["examples"].tolist() + eae_random_any["examples"].tolist(),
    eae_none["argument_f1"].tolist() + eae_random_any["argument_f1"].tolist(),
    color="blue",
    linestyle=":",
    linewidth=1.2,
    label=None,
)
ax2.plot(
    eae_none["examples"].tolist() + eae_random_same["examples"].tolist(),
    eae_none["argument_f1"].tolist() + eae_random_same["argument_f1"].tolist(),
    color="green",
    linestyle=":",
    linewidth=1.2,
    label=None,
)
ax2.plot(
    eae_none["examples"].tolist() + eae_close_any["examples"].tolist(),
    eae_none["argument_f1"].tolist() + eae_close_any["argument_f1"].tolist(),
    color="#c947ba",
    linestyle=":",
    linewidth=1.2,
    label=None,
)
ax2.plot(
    eae_none["examples"].tolist() + eae_close_same["examples"].tolist(),
    eae_none["argument_f1"].tolist() + eae_close_same["argument_f1"].tolist(),
    color="orange",
    linestyle=":",
    linewidth=1.2,
    label=None,
)

ax2.text(
    0.03,
    0.97,
    "Argument Extraction\nMistral 24B",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
)

ax2.set_ylabel("Argument F1", fontsize=12)
ax2.set_xlabel("Number of examples", fontsize=12)
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(ax1.get_xlim())
ax2.grid(axis="y")
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(axis="y", direction="in", which="both", left=True, right=True)
ax2.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.grid(axis="x")

ax2.legend(loc="lower right")

# Add subplot labels
ax1.text(
    -0.1,
    1.0,
    "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax2.text(
    -0.1,
    1.0,
    "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)

# Save the figure
fig.savefig(this_file.parent / f"{this_file.stem}.pdf")
