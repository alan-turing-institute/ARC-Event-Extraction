from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from glob import glob

import pandas as pd
from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)
from tqdm import tqdm

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_through_files(
    golds: dict[str, dict | list],
    process_func: Callable[
        [dict[str, dict | list], dict[str, dict | list], bool], dict[str, int | float]
    ],
    id_key: str,
    output_folder: str,
    save_name: str,
    semi_compare: bool = False,
) -> None:
    results_df = pd.DataFrame()

    for file in tqdm(sorted(glob(os.path.join(MAIN_DIR, output_folder, "*.jsonl")))):
        outputs = load_events(file, id_key)
        if outputs is None:
            continue
        file_name = os.path.basename(file)[: -len(".jsonl")]
        stats = process_func(golds, outputs, semi=semi_compare)

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        "file_name": [file_name],
                        **stats,
                    }
                ),
            ],
            ignore_index=True,
        )

    os.makedirs(os.path.dirname(os.path.join(MAIN_DIR, save_name)), exist_ok=True)

    results_df.to_csv(os.path.join(MAIN_DIR, save_name), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics for event extraction."
    )
    parser.add_argument(
        "output_folder", type=str, help="Folder containing the output files."
    )
    parser.add_argument("save_loc", type=str, help="Location to save the statistics.")
    parser.add_argument(
        "--semi_compare",
        action="store_true",
        help="Use partial string match for comparison.",
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

    run_through_files(
        eae_golds,
        stats_on_argument_detection,
        "evt_id",
        args.output_folder,
        args.save_loc + "_eae.csv",
        semi_compare=args.semi_compare,
    )
    run_through_files(
        ed_golds,
        stats_on_event_detection,
        "wnd_id",
        args.output_folder,
        args.save_loc + "_ed.csv",
        semi_compare=args.semi_compare,
    )


if __name__ == "__main__":
    main()
