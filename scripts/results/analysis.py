from __future__ import annotations

import argparse

import pandas as pd
import wandb
import yaml
from arcee.constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from numpy import logical_and


def extract_run_information(run: wandb.apis.public.runs.Run) -> dict[str, str] | None:
    substrings = run.name.split("_")
    run_info = {
        "folder": run.config["folder"],
        "dataset": substrings[0],
        "split": substrings[1],
        "model": substrings[2],
        "prompt": substrings[3],
        "constraint": substrings[4],
    }
    num_example = 0
    if (len(substrings)) > 5:
        num_example = int(substrings[5])
    return {
        **run_info,
        "num_example": num_example,
    }


def run_to_row(run: wandb.apis.public.runs.Run) -> dict[str, str | float]:
    run_info = extract_run_information(run)
    stats = {
        stat: value
        for stat, value in run.summary.items()
        if stat in ["full", "partial"]
    }
    return {
        **run_info,
        **stats,
    }


def runs_to_df(runs: list[wandb.apis.public.runs.Run]) -> pd.DataFrame:
    return pd.DataFrame([run_to_row(run) for run in runs])


def prepare_cfg(cfg: dict, job_type: str) -> dict:
    new_cfg = {key: value for key, value in cfg.items() if key not in ["ed", "eae"]}
    return {**new_cfg, **cfg[job_type]}


def select_runs(
    cfg: dict,
    runs: list[wandb.apis.public.runs.Run],
    job_type: str,
) -> pd.DataFrame:
    if not any([job_type in ["ed", "eae"]]):
        msg = f"job_type must be one of 'ed' or 'eae' but is is {job_type}"
        raise ValueError(msg)
    if job_type == "ed":
        filter_column = "full/event_f1"
    if job_type == "eae":
        filter_column = "full/argument_f1"
    selected_runs = [run for run in runs if filter_column in run.summary]
    runs_df = runs_to_df(selected_runs)
    job_cfg = prepare_cfg(cfg, job_type)
    return runs_df.loc[
        logical_and.reduce([runs_df[key].isin(job_cfg[key]) for key in job_cfg])
    ]


def get_dfs(cfg: dict, runs: list[wandb.apis.public.runs.Run]) -> None:
    ed_df = select_runs(cfg, runs, "ed")
    eae_df = select_runs(cfg, runs, "eae")
    return ed_df, eae_df


def main(cfg: dict):
    client = wandb.Api()
    runs = client.runs(
        path=f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}",
        filters={"jobType": "stats"},
    )
    runs = [run for run in runs if run.name[0:6] != "failed"]
    runs = [run for run in runs if run.config["folder"] in cfg["folder"]]
    del cfg["folder"]
    ed_df, eae_df = get_dfs(cfg, runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis for desired statistics.")
    parser.add_argument(
        "config", type=str, help="Config specifying which stats jobs to analyse."
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
