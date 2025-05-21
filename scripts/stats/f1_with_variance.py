"""
This script calculates the F1 score and its variance for a given set of predictions
and ground truth labels.

The script uses a YAML configuration file to specify the following parameters:
- `task`: The task type, either 'eae' (event argument extraction)
    or 'ed' (event detection).
- `gold_location`: Path to the gold standard data file.
- `bootstrap_count`: Number of bootstrap samples to generate (default is 100).
- `seed`: Random seed for reproducibility (optional).
- `save_location`: Path to save the results as a CSV file (optional).
- `semi_string`: Boolean indicating whether to use semi-string matching (optional).
- `experiments`: A dictionary of experiments, where each key is the experiment name
    and the value is a dictionary containing:
  - `data_loc`: Path to the experiment's data file.
  - Additional metadata fields (e.g., model, prompt, fine-tune status, etc.), which
    will be included in the output CSV file.
"""

from __future__ import annotations

import argparse  # Add import for argparse
import os

import numpy as np
import pandas as pd
import yaml
from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)
from tqdm import tqdm


def bootstrap_sample(data: dict[str, any], rng: np.random.Generator) -> dict[str, any]:
    """
    Generate a bootstrap sample from the given dictionary.

    Args:
        data (dict[str, any]): The original dictionary to sample from.
        rng (np.random.Generator): A random number generator.

    Returns:
        dict[str, any]: A new dictionary sampled with replacement.
    """
    keys = list(data.keys())
    sampled_keys = rng.choice(keys, size=len(keys), replace=True)
    return {key: data[key] for key in sampled_keys}


def strip_by_events(
    data: dict[str, any],
    gold_data: dict[str, any],
    min_events: int | None = None,
    max_events: int | None = None,
) -> None:
    """
    Filter the data dictionary based on the number of events in the gold data.

    Args:
        data (dict[str, any]): The data dictionary to filter.
        gold_data (dict[str, any]): The gold data dictionary to compare against.
        min_events (int | None): Minimum number of events to keep (default: None).
        max_events (int | None): Maximum number of events to keep (default: None).

    Returns:
        filtered_data (dict[str, any]): The filtered data dictionary.
    """
    if min_events is not None:
        data = {k: v for k, v in data.items() if len(gold_data[k]) >= min_events}
    if max_events is not None:
        data = {k: v for k, v in data.items() if len(gold_data[k]) <= max_events}

    return data


def main(config_loc: str) -> None:
    main_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    with open(config_loc) as f:
        config = yaml.safe_load(f)

    metric = config.get("metric", None)

    task = config.get("task")
    if task == "eae":
        id_key = "evt_id"
        stats_function = stats_on_argument_detection
        metric = "argument_f1" if metric is None else metric
    elif task == "ed":
        id_key = "wnd_id"
        stats_function = stats_on_event_detection
        metric = "event_f1" if metric is None else metric
    else:
        msg = f"The task provided was {task}, but should be one of 'eae' or 'ed'."
        raise ValueError(msg)

    gold_loc = config["gold_location"]
    gold_loc = os.path.join(main_dir, gold_loc)
    if not os.path.exists(gold_loc):
        msg = f"The data file at '{gold_loc}' does not exist."
        raise FileNotFoundError(msg)

    gold_data = load_events(gold_loc, id_key)  # Load the gold data to compare against

    bootstrap_count = config.get("bootstrap_count", 100)
    if not isinstance(bootstrap_count, int):
        msg = f"The 'bootstrap_count' must be an integer was {bootstrap_count}."
        raise ValueError(msg)

    seed = config.get("seed")
    if not (seed is None or isinstance(seed, int)):
        msg = "The 'seed' value in the config must be an integer or None."
        raise ValueError(msg)

    semi = config.get("semi_string", False)

    exp_dict = config.get("experiments")
    if not isinstance(exp_dict, dict):
        msg = "Please provide a dictionary of experiments."
        raise ValueError(msg)
    if not all(isinstance(k, str) for k in exp_dict):
        msg = "The experiment names must be strings."
        raise ValueError(msg)
    if not all(isinstance(v, dict) for v in exp_dict.values()):
        msg = "The experiment values must be dictionaries."
        raise ValueError(msg)
    if not all("data_loc" in v for v in exp_dict.values()):
        msg = "The experiments must contain 'data_loc'."
        raise ValueError(msg)

    exp_data: dict = {}  # Dictionary to hold experiment data
    exp_info: dict = {}  # Dictionary to hold experiment metadata
    for exp_name, exp_vals in exp_dict.items():
        data_loc = os.path.join(main_dir, exp_vals["data_loc"])
        if not os.path.exists(data_loc):
            msg = f"The file for {exp_name} at '{exp_vals['data_loc']}' does not exist."
            raise FileNotFoundError(msg)

        exp_data[exp_name] = load_events(data_loc, id_key)

        if exp_data[exp_name] is None:
            msg = (
                f"The file for {exp_name} at '{exp_vals['data_loc']}' is empty."
                "Could you have mixed up the event and argument detection files?"
            )
            raise ValueError(msg)

        # Event count filtering
        min_events = exp_vals.get("min_events")
        max_events = exp_vals.get("max_events")
        if task == "ed" and (min_events or max_events):
            exp_data[exp_name] = strip_by_events(
                exp_data[exp_name], gold_data, min_events, max_events
            )

        exp_info[exp_name] = exp_vals
        exp_info[exp_name].pop("data_loc", None)

    rng = np.random.default_rng(seed)

    for e_name, e_data in tqdm(exp_data.items()):
        exp_info[e_name][metric] = stats_function(gold_data, e_data, semi=semi)[metric]

        # Perform bootstrapping
        bootstrap_results = []
        for _ in range(bootstrap_count):
            sampled_data = bootstrap_sample(e_data, rng)
            stats = stats_function(gold_data, sampled_data, semi=semi)
            bootstrap_results.append(stats[metric])

        exp_info[e_name]["bs_mean"] = np.mean(bootstrap_results)
        exp_info[e_name]["bs_var"] = np.var(bootstrap_results)
        exp_info[e_name]["bs_std"] = np.std(bootstrap_results)

    # Save Results
    results_df = pd.DataFrame.from_dict(exp_info, orient="index")
    print(results_df)

    save_loc = config.get("save_location", None)
    if save_loc:
        results_df.to_csv(save_loc, index=True, index_label="experiment")
        print(f"Results saved to {save_loc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 score with standard error.")
    parser.add_argument("config_loc", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    print(f"Config location: {args.config_loc}")
    main(args.config_loc)
