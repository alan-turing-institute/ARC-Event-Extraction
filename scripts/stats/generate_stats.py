from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from glob import glob

import pandas as pd
import wandb
import yaml
from arcee.constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)
from arcee.utils import prepare_wandb_name
from tqdm import tqdm

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def layout_config(to_read: dict, to_append: dict) -> None:
    """RECURSION"""
    for key, value in to_read.items():
        if isinstance(value, dict):
            layout_config(value, to_append)
        else:
            to_append[key] = value


def run_through_files(
    golds: dict[str, dict | list],
    process_func: Callable[
        [dict[str, dict | list], dict[str, dict | list], bool], dict[str, int | float]
    ],
    id_key: str,
    output_folder: str,
    save_name: str,
    semi_compare: bool = False,
    use_wandb: bool = True,
    dataset: str | None = None,
    skip_config: bool = False,
) -> None:
    results_df = pd.DataFrame()

    print(skip_config)

    for file in tqdm(sorted(glob(os.path.join(MAIN_DIR, output_folder, "*.jsonl")))):
        outputs = load_events(file, id_key)
        if outputs is None:
            continue

        file_name = os.path.basename(file)[: -len(".jsonl")]

        # Optionally use weights and biases
        if use_wandb:
            run_name = prepare_wandb_name(file_name)
            run = wandb.init(
                entity=WANDB_ENTITY_NAME,
                project=WANDB_PROJECT_NAME,
                name=run_name,
                job_type="stats",
                config={
                    "file_name": file_name,
                    "job_type": "stats",  # legacy reasons
                    "semi_compare": semi_compare,
                    "dataset": dataset,
                },
            )

        config_vals = {}
        if not skip_config:
            con_file = file.replace(
                os.path.join(MAIN_DIR, "outputs"),
                os.path.join(MAIN_DIR, "scripts/configs"),
            )
            con_file = con_file.replace(".jsonl", ".yaml")
            with open(con_file) as f:
                config_whole = yaml.safe_load(f)

            layout_config(config_whole, config_vals)

        stats = process_func(golds, outputs, semi=semi_compare)
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        "file_name": [file_name],
                        **config_vals,
                        **stats,
                    }
                ),
            ],
            ignore_index=True,
        )

        if use_wandb:
            run.log(stats)
            run.finish()

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
        "--skip_config",
        action="store_true",
        help="If to include the config info in the table.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ACE05",
        help="Dataset to use for comparison.",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Whether to log stats to weights and biases",
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
        use_wandb=args.use_wandb,
        dataset=args.dataset,
        skip_config=args.skip_config,
    )
    run_through_files(
        ed_golds,
        stats_on_event_detection,
        "wnd_id",
        args.output_folder,
        args.save_loc + "_ed.csv",
        semi_compare=args.semi_compare,
        use_wandb=args.use_wandb,
        dataset=args.dataset,
        skip_config=args.skip_config,
    )


if __name__ == "__main__":
    main()
