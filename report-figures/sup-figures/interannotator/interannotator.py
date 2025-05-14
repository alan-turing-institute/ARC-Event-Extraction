from __future__ import annotations

import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

main_dir = os.path.abspath(os.path.join(__file__, "../../../.."))

fig = Figure(figsize=(8, 5), tight_layout=True)
axis = fig.subplots(1, 1)

accuracy = np.arange(
    0.1,
    1.005,
    0.01,
)

one_bad_ia = accuracy
one_bad_max = 1 - (1 - accuracy) / 2

worse_ia = accuracy
worse_max = accuracy


def make_agree_max(choice, hard):
    easy = 1 - hard
    good_rate = np.arange(0.1, 1.005, 0.01)
    bad_rate = 1 - good_rate
    easy_ia = good_rate * good_rate + (1 - good_rate * good_rate) / choice
    easy_max = good_rate + bad_rate / choice
    hard_ia = 1 / choice
    hard_max = 1 / choice
    ia = easy * easy_ia + hard * hard_ia
    max = easy * easy_max + hard * hard_max
    return ia, max


axis.plot(one_bad_ia, one_bad_max, label="One Bad Annotator")

axis.plot(*make_agree_max(5, 0), ls=(0, (3, 1, 1, 1, 1, 1)), label="Random 5 Choice")
axis.plot(*make_agree_max(2, 0), "--", label="Random 2 Choice")

axis.plot(*make_agree_max(2, 0.2), "-.", label="2 Choice 20% Hard")
axis.plot(*make_agree_max(5, 0.2), ":", label="5 Choice 20% Hard")

axis.plot(worse_ia, worse_max, label="Same Mistakes")

axis.set_xlim(0.45, 1)
axis.set_ylim(0.5, 1)

axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)
axis.minorticks_on()

axis.xaxis.set_major_formatter(PercentFormatter(1.0))
axis.yaxis.set_major_formatter(PercentFormatter(1.0))

axis.grid()
axis.legend(loc="lower right")
axis.set_xlabel("Inter-annotator agreement")
axis.set_ylabel("Maximum achievable score")

fig.savefig(
    os.path.join(
        main_dir, "report-figures/sup-figures/interannotator/interannotator.pdf"
    )
)
