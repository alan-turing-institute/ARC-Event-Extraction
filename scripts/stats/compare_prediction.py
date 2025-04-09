from __future__ import annotations

import argparse
import os
from itertools import product

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)
from tqdm import tqdm


def plot_answer_matrix(
    matrix: np.ndarray,
    title: str,
    model_names: list[str],
    file_name: str,
) -> None:
    """
    Plot the confusion matrix based on the given confusion values and source names. The
    figure is shown and also saved to the given file name.

    Args:
        matrix (np.ndarray): The confusion matrix.
        title (str): The title of the plot.
        model_names (list[str]): The names of the models.
        file_name (str): The name of the file to save the plot
    """
    number_of_classes = len(model_names)
    # Plotting the matrix
    plt.figure(dpi=300)
    ax = plt.subplot(111)
    ax.matshow(matrix.astype(float), cmap=cc.cm.kbc_r)

    # Setting labels for the axes
    ax.set_xlabel("Predicted answers")
    ax.set_ylabel("Referenced answers")
    ax.set_title(f"{title}")

    # Adding text in each cell
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f"{100*val:.0f}", ha="center", va="center", color="black")

    # Setting tick marks for models
    ax.set_xticks(np.arange(number_of_classes))
    ax.set_yticks(np.arange(number_of_classes))
    ax.set_xticklabels(model_names, rotation=30)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticklabels(model_names)

    plt.savefig(file_name, bbox_inches="tight")


def parse_args():
    """
    Parse command-line arguments to get the YAML config file path.
    """
    parser = argparse.ArgumentParser(
        description="Compare predictions using a YAML config file."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.
    """
    with open(config_path) as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()
    config = load_config(args.config)

    mair_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    output_dict = config.get("output_dict")
    if output_dict is None:
        msg = "Please provide a dictionary of output paths."
        raise ValueError(msg)

    task = config["task"]

    if task == "eae":
        func_choice = stats_on_argument_detection
        id_key = "evt_id"
    elif task == "ed":
        func_choice = stats_on_event_detection
        id_key = "wnd_id"
    else:
        msg = f"The task provided was {task}, but should be one of 'eae' or 'ed'."
        raise ValueError(msg)

    metric = config["metric"]

    semi_string = config.get("semi_string", False)

    compare_df = pd.DataFrame(columns=output_dict.keys())

    for (key1, value1), (key2, value2) in tqdm(
        product(output_dict.items(), repeat=2), total=len(output_dict) ** 2
    ):
        e1 = load_events(os.path.join(mair_dir, value1), id_key)
        e2 = load_events(os.path.join(mair_dir, value2), id_key)

        compare_df.loc[key1, key2] = func_choice(
            e1, e2, semi=semi_string, real_gold=False
        )[metric]

    print(compare_df)
    plot_answer_matrix(
        compare_df.values,
        config.get("plot_title", "Compare Predictions"),
        list(output_dict.keys()),
        config.get("output_plot_path", "stats/figures/default_figure_name.png"),
    )

    save_loc = config.get("output_csv_path")
    if save_loc:
        save_loc = os.path.join(mair_dir, save_loc)
        compare_df.to_csv(save_loc, index=False)
        print(f"Results saved to {save_loc}")


if __name__ == "__main__":
    main()
