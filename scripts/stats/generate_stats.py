from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from glob import glob

import wandb
from arcee.constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)
from arcee.utils import prepare_wandb_name
from tqdm import tqdm

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_stats(
    outputs: dict[str, dict | list],
    golds: dict[str, dict | list],
    keep_whitespace: bool,
    process_func: Callable[
        [dict[str, dict | list], dict[str, dict | list], bool], dict[str, int | float]
    ],
) -> dict[str, float]:
    stats_full = process_func(
        golds, outputs, semi=False, keep_whitespace=keep_whitespace
    )
    stats_full = {f"full/{key}": value for key, value in stats_full.items()}
    stats_partial = process_func(
        golds, outputs, semi=True, keep_whitespace=keep_whitespace
    )
    stats_partial = {f"partial/{key}": value for key, value in stats_partial.items()}
    return {**stats_full, **stats_partial}


def run_through_files(
    golds: dict[str, dict | list],
    process_func: Callable[
        [dict[str, dict | list], dict[str, dict | list], bool], dict[str, int | float]
    ],
    id_key: str,
    output_folder: str,
    dataset: str | None = None,
    keep_whitespace: bool = False,
    event_filter: list[str] | None = None,
) -> None:
    for file in tqdm(sorted(glob(os.path.join(MAIN_DIR, output_folder, "*.jsonl")))):
        outputs = load_events(file, id_key)
        if outputs is None:
            continue

        file_name = os.path.basename(file)[: -len(".jsonl")]

        # Use weights and biases
        run_name = prepare_wandb_name(file_name)
        run = wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            name=run_name,
            job_type="stats",
            config={
                "file_name": file_name,
                "job_type": "stats",  # legacy reasons
                "dataset": dataset,
                "keep_whitespace": keep_whitespace,
            },
        )

        stats = compute_stats(outputs, golds, keep_whitespace, process_func)

        if event_filter is not None:
            filtered_outputs = {
                key: value for key, value in outputs.items() if key in event_filter
            }
            filtered_stats = compute_stats(
                filtered_outputs, golds, keep_whitespace, process_func
            )
            filtered_stats = {
                f"filtered_{key}": value for key, value in filtered_stats.items()
            }
            stats = {**stats, **filtered_stats}

        run.log(stats)
        run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics for event extraction."
    )
    parser.add_argument(
        "output_folder", type=str, help="Folder containing the output files."
    )
    parser.add_argument(
        "--keep_whitespace",
        action="store_true",
        help="If to keep the whitespace before comparison.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ACE05",
        help="Dataset to use for comparison.",
    )
    args = parser.parse_args()

    gold_arg_file = f"data/{args.dataset}/gold/gold_label_args.jsonl"
    gold_trig_file = f"data/{args.dataset}/gold/gold_label_triggers.jsonl"

    eae_golds = load_events(os.path.join(MAIN_DIR, gold_arg_file), "evt_id")
    ed_golds = load_events(os.path.join(MAIN_DIR, gold_trig_file), "wnd_id")
    ed_filter = [key for key, value in ed_golds.items() if value != []]

    run_through_files(
        eae_golds,
        stats_on_argument_detection,
        "evt_id",
        args.output_folder,
        dataset=args.dataset,
        keep_whitespace=args.keep_whitespace,
    )
    run_through_files(
        ed_golds,
        stats_on_event_detection,
        "wnd_id",
        args.output_folder,
        dataset=args.dataset,
        keep_whitespace=args.keep_whitespace,
        event_filter=ed_filter,
    )


if __name__ == "__main__":
    main()
