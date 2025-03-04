from __future__ import annotations

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

GOLD_ARG_FILE = "data/ACE05/gold/gold_label_args.jsonl"
GOLD_TRIG_FILE = "data/ACE05/gold/gold_label_triggers.jsonl"

OUTPUT_FOLDER = "outputs/ollama/first_run/"

SAVE_LOC = "stats/ollama/first_run"


main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_through_files(
    golds: dict[str, dict | list],
    process_func: Callable[
        [dict[str, dict | list], dict[str, dict | list]], dict[str, int | float]
    ],
    id_key: str,
    output_folder: str,
    save_name: str,
) -> None:
    results_df = pd.DataFrame()

    for file in tqdm(sorted(glob(os.path.join(main_dir, output_folder, "*.jsonl")))):
        outputs = load_events(file, id_key)
        if outputs is None:
            continue
        file_name = os.path.basename(file)[: -len(".jsonl")]
        stats = process_func(golds, outputs)

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

    os.makedirs(os.path.dirname(os.path.join(main_dir, save_name)), exist_ok=True)

    results_df.to_csv(os.path.join(main_dir, save_name), index=False)


def main():
    eae_golds = load_events(os.path.join(main_dir, GOLD_ARG_FILE), "evt_id")
    ed_golds = load_events(os.path.join(main_dir, GOLD_TRIG_FILE), "wnd_id")

    run_through_files(
        eae_golds,
        stats_on_argument_detection,
        "evt_id",
        OUTPUT_FOLDER,
        SAVE_LOC + "_eae.csv",
    )
    run_through_files(
        ed_golds,
        stats_on_event_detection,
        "wnd_id",
        OUTPUT_FOLDER,
        SAVE_LOC + "_ed.csv",
    )


if __name__ == "__main__":
    main()
