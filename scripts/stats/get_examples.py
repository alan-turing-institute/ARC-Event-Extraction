from __future__ import annotations

import argparse  # Add import for argparse
import json  # Add import for JSON
import os

import numpy as np
import yaml
from arcee.stats.gold_compare import load_events


def compare_list_items(lst: list, task: str, size: int) -> bool:
    if len(lst) < 1:
        return True
    if task == "ed":
        lst = [sorted(x, key=str) for x in lst]
    return len(set(lst)) == size

def all_items_equal(lst: list, task: str):
    return compare_list_items(lst, task, 1)

def all_items_unequal(lst: list, task: str):
    return compare_list_items(lst, task, len(lst))


def main(config_loc: str) -> None:
    main_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    with open(config_loc) as f:
        config = yaml.safe_load(f)

    output_dict = config.get("output_dict")
    if not isinstance(output_dict, dict):
        msg = "Please provide a dictionary of output paths."
        raise ValueError(msg)

    agree = config.get("agree", [])
    disagree = config.get("disagree", [])

    if not isinstance(agree, list):
        msg = "The 'agree' parameter must be a list."
        raise ValueError(msg)
    if not isinstance(disagree, list):
        msg = "The 'disagree' parameter must be a list."
        raise ValueError(msg)

    for list_name, items in [("agree", agree), ("disagree", disagree)]:
        for item in items:
            if not isinstance(item, str):
                msg = f"Item '{item}' in '{list_name}' must be a string."
                raise ValueError(msg)
            if item not in output_dict:
                msg = (
                    f"Item '{item}' in '{list_name}' must correspond"
                    f" to a key in 'output_dict'."
                )
                raise ValueError(msg)

    data_loc = os.path.join(main_dir, config["data_location"])

    with open(data_loc, "r") as f:
        pulled_data = [json.loads(line) for line in f]

    task = config.get("task")
    if task == "eae":
        id_key = "evt_id"
        text_data = {}
        trigger_data: dict[str, str] = {}
        for item in pulled_data:
            for event in item.get("event_mentions", []):
                event_id = event["id"]
                text_data[event_id] = item["text"]
                trigger_data[event_id] = (
                    f"{event['event_type']} - '{event['trigger']['text']}'"
                )
    elif task == "ed":
        id_key = "wnd_id"
        text_data = {item["wnd_id"]: item["text"] for item in pulled_data}
    else:
        msg = f"The task provided was {task}, but should be one of 'eae' or 'ed'."
        raise ValueError(msg)

    answer_dict = {
        name: load_events(os.path.join(main_dir, out_loc), id_key)
        for name, out_loc in output_dict.items()
    }

    seed = config.get("seed")
    if not (seed is None or isinstance(seed, int)):
        msg = "The 'seed' value in the config must be an integer or None."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)

    index_set = set()
    for answers in answer_dict.values():
        index_set = (
            index_set.intersection(set(answers.keys()))
            if index_set
            else set(answers.keys())
        )

    shuffled_indexes = sorted(index_set)  # Sort so seed is reproducible
    rng.shuffle(shuffled_indexes)

    max_examples = config.get("max_examples", 10)
    if not isinstance(max_examples, int):
        msg = f"The 'max_examples' must be an integer was {max_examples}."
        raise ValueError(msg)

    print("++++++++++++++++++++++++")
    print("++++++++++++++++++++++++")
    print("EXAMPLES")
    for index in shuffled_indexes:
        agree_values = [answer_dict[name][index] for name in agree]
        if not all_items_equal(agree_values, task):
            continue
        if (
            task == "eae"
            and len(agree_values) != 0
            and all(val is None for val in agree_values[0].values())
        ):
            continue
        if task == "ed" and len(agree_values) != 0 and len(agree_values[0]) == 0:
            continue

        disagree_values = [answer_dict[name][index] for name in disagree]
        if not all_items_unequal(disagree_values, task):
            continue

        print(f"Index: {index}")
        if task == "eae":
            print(f"Text: {text_data[index]}")
            print(f"Trigger: {trigger_data[index]}")
        else:
            print(f"Text: {text_data[index]}")
        for name, answers in answer_dict.items():
            print(f"{name}: {answers[index]}")

        max_examples -= 1
        if max_examples <= 0:
            print("++++++++++++++++++++++++")
            print("++++++++++++++++++++++++")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give examples for event extraction.")
    parser.add_argument("config_loc", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    print(f"Config location: {args.config_loc}")
    main(args.config_loc)
