from __future__ import annotations

import argparse

import wandb
from arcee.config.experiment import ExperimentConfig
from arcee.constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from arcee.eval.evaluator import Evaluator
from arcee.utils import prepare_wandb_name


def main(config: ExperimentConfig, wandb_job_name: str) -> None:
    # Initialise wandb run
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        name=wandb_job_name,
        job_type="inference",
        config={
            "job_type": "inference",
            **config.to_dict(),
        },
    )

    # Initialise evaluator
    evaluator = Evaluator(
        output_path=config.output_path,
        wandb_run=run,
        **config.data_kwargs.to_dict(),
        **config.model_kwargs.to_dict(),
        **config.prompt_kwargs.to_dict(),
    )

    # Evaluate
    evaluator.evaluate()

    # Finish run
    run.finish()


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser(description="Get arguments for evaluating output")
    parser.add_argument(
        "--config_path",
        required=True,
        help="String containing the local path to the experiment config",
    )
    parser.add_argument(
        "--ollama_port",
        required=False,
        type=int,
        help="Port to use for OLLAMA",
    )
    args = parser.parse_args()

    # Init experiment config
    config = ExperimentConfig.from_yaml(args.config_path)
    output_path = args.config_path.replace("yaml", "jsonl")
    output_path = output_path.replace("scripts/configs", "outputs")
    config.output_path = output_path
    if args.ollama_port:
        config.model_kwargs.model_init_kwargs["host_port"] = args.ollama_port

    # Job name
    wandb_job_name = args.config_path.replace("scripts/configs/", "").replace(
        ".yaml", ""
    )
    wandb_job_name = prepare_wandb_name(wandb_job_name)

    # Run
    main(config, wandb_job_name)
