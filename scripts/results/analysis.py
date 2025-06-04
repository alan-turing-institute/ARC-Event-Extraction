from __future__ import annotations

import argparse
import os

import pandas as pd
import wandb
import yaml
from arcee.constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from numpy import int64, logical_and, logical_or

#### Constants -------------------------------------------------------------------------

COLUMN_TO_LABEL_MAP = {
    "model": "Model",
    "num_example": "N. Examples",
    "rank": "LoRA Rank",
    "full/event_f1": "F1",
    "full/event_recall": "Recall",
    "full/event_precision": "Precision",
    "partial/event_f1": "Partial F1",
    "full/event_type_f1": "Event Type F1",
    "full/argument_f1": "F1",
    "full/argument_recall": "Recall",
    "full/argument_precision": "Precision",
    "partial/argument_f1": "Partial F1",
    "full/event_accuracy": "Accuracy",
}


MODEL_SIZE_MAP = {
    "llama3b": 3,
    "llama8b": 8,
    "phi4": 14,
    "mistral24b": 24,
    "gemma27b": 27,
    "llama70b": 70,
    "mistral123b": 123,
    "gpt4o": 999999999,
}


PROMPT_LABEL_MAP = {
    "EAE1": "EAE No Examples",
    "EAE3": "EAE Random Examples",
    "EAE6": "EAE Similar Examples",
    "EAE8": "EAE Similar Examples Same Event Type",
    "ED1": "ED No Examples",
    "ED3": "ED Random Examples",
    "ED4": "ED Similar Examples",
    "ED8": "ED Similar Examples Same Event Type",
}


APPROACH_LABEL_MAP = {"icl": "In-Context Learning", "ft": "Fine-Tuned"}

#### Wandb Processing Functions --------------------------------------------------------


def extract_run_information(run: wandb.apis.public.runs.Run) -> dict[str, str] | None:
    substrings = run.name.split("_")
    print(substrings)
    run_info = {
        "folder": run.config["folder"],
        "dataset": substrings[0],
        "split": substrings[1],
        "model": substrings[2],
        "prompt": substrings[3],
        "constraint": substrings[4],
    }
    if run_info["folder"] != "fineTunedInference":
        num_example = 0
        if (len(substrings)) > 5:
            try:
                num_example = int(substrings[5])
            except ValueError:
                num_example = 0
        return {
            **run_info,
            "num_example": num_example,
        }
    num_example = 0
    if (len(substrings)) > 6:
        num_example = int(substrings[6])
    try:
        rank = int(substrings[5][-4:-1])
    except ValueError:
        rank = int(substrings[5][-3:-1])
    return {**run_info, "num_example": num_example, "rank": rank}


def run_to_row(run: wandb.apis.public.runs.Run) -> dict[str, str | float]:
    run_info = extract_run_information(run)
    stats = {
        stat: value
        for stat, value in run.summary.items()
        if "full" in stat or "partial" in stat
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
    runs_df = runs_df.loc[
        ~runs_df["model"].isin(["gpt4omini", "o3mini"])
    ]  # for GPT jobs
    runs_df["size"] = runs_df["model"].map(lambda x: MODEL_SIZE_MAP[x])
    sort_by = ["size", "prompt", "num_example"]
    if "rank" in runs_df.columns:
        sort_by.insert(1, "rank")
    runs_df = runs_df.sort_values(by=sort_by)
    return runs_df.loc[
        logical_and.reduce([runs_df[key].isin(job_cfg[key]) for key in job_cfg])
    ]


def get_dfs(cfg: dict, runs: list[wandb.apis.public.runs.Run]) -> None:
    ed_df = select_runs(cfg, runs, "ed")
    eae_df = select_runs(cfg, runs, "eae")
    return ed_df, eae_df


#### LaTeX Functions -------------------------------------------------------------------


def latex_bold(tex: str) -> str:
    return "\\textbf{" + tex + "}"


def latex_round(number: float) -> str:
    return f"{number:.2f}"


def latex_float(number: float, df: pd.DataFrame, index: int) -> str:
    if number == df[df.columns[index]].max():
        return latex_bold(latex_round(number))
    return latex_round(number)


def df_to_tex(df: pd.DataFrame, t_name: str, t_label: str) -> str:
    nrow, ncol = df.shape
    labels = [COLUMN_TO_LABEL_MAP[col] for col in df.columns]
    tab_head = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{t_name}}}\n"
        f"\\label{{tab:{t_label}}}\n"
        "\\setlength\\tabcolsep{1.5pt}\n"
        f"\\begin{{tabular}}{{c|{''.join(['c' for _ in range(1,ncol)])}}}\n"
    )
    for i in range(ncol):
        line_end = " & "
        if i == (ncol - 1):
            line_end = " \\\\\n\\hline\n"
        tab_head = tab_head + f"\\textbf{{{labels[i]}}}" + line_end
    tab_contents = ""
    for i in range(nrow):
        tab_contents = (
            tab_contents
            + " & ".join(
                [
                    str(x) if type(x) in [str, int, int64] else latex_float(x, df, i)
                    for i, x in enumerate(df.iloc[i])
                ]
            )
            + "\\\\\n"
        )
    tab_tail = (
        "\\end{tabular}\n"
        # "\\caption*{\\\\\\textit{Values presented with bootstrapped 95\\% "
        # "confidence intervals.\\\\\nLargest value for each row presented in bold.}}"
        "\n\\end{table}"
    )
    return tab_head + tab_contents + tab_tail


#### Output Functions ------------------------------------------------------------------


def df_to_output(df: pd.DataFrame, selection: list[str], file: str) -> None:
    for prompt, sub_df in df.groupby("prompt"):
        if ("rank" in df.columns) and ("rank" not in selection):
            selection.insert(0, "rank")
        if prompt in ["ED1", "EAE1"]:
            col_selection = ["model", *selection]
        else:
            col_selection = ["model", "num_example", *selection]
        out_df = sub_df[col_selection]
        tex = df_to_tex(
            out_df,
            t_name=f"{APPROACH_LABEL_MAP[file]}: {PROMPT_LABEL_MAP[prompt]}",
            t_label=f"tab:{file}-{prompt}",
        )
        with open(f"tables/results_{file}_{prompt}.tex", "w") as f:
            f.write(tex)


def ed_df_to_output(df: pd.DataFrame, file: str) -> None:
    df_to_output(
        df=df,
        selection=[
            "full/event_f1",
            "full/event_recall",
            "full/event_precision",
            "partial/event_f1",
            "full/event_type_f1",
        ],
        file=file,
    )


def eae_df_to_output(df: pd.DataFrame, file: str) -> None:
    df_to_output(
        df=df,
        selection=[
            "full/argument_f1",
            "full/argument_recall",
            "full/argument_precision",
            "partial/argument_f1",
            "full/event_accuracy",
        ],
        file=file,
    )


#### Data Selection Functions ----------------------------------------------------------


def select_data_from_group(runs, cfg_group):
    runs = [run for run in runs if run.config["folder"] in cfg_group["folder"]]
    del cfg_group["folder"]
    return get_dfs(cfg_group, runs)


def select_data(runs, cfg):
    ed_dfs, eae_dfs = [], []
    for cfg_group in cfg["run_groups"]:
        ed_df, eae_df = select_data_from_group(runs, cfg_group)
        ed_dfs.append(ed_df)
        eae_dfs.append(eae_df)
    return pd.concat(ed_dfs), pd.concat(eae_dfs)


#### Main ------------------------------------------------------------------------------


def main(cfg: dict):
    client = wandb.Api()
    runs = client.runs(
        path=f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}",
        filters={
            "jobType": "stats",
            "state": "finished",
        },
    )
    runs = [run for run in runs if run.name[0:6] != "failed"]
    name = cfg["name"]
    if name == "ft":
        runs = [run for run in runs if "nodefs" not in run.name]
        valid = [
            "A4bEDOEm32w",
            "A4bEDOEm64w",
            "A4bEAEm32w",
            "A4bEAEm64w",
            "A4bEAEm128w",
            "A4bEAEm256w",
            "A4bEAEm512w",
        ]
        runs = [
            run for run in runs if logical_or.reduce([x in run.name for x in valid])
        ]
    ed_df, eae_df = select_data(runs, cfg)
    ed_df_to_output(ed_df, name)
    eae_df_to_output(eae_df, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis for desired statistics.")
    parser.add_argument(
        "config", type=str, help="Config specifying which stats jobs to analyse."
    )
    if not os.path.exists("tables"):
        os.path.mkdir("tables")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
