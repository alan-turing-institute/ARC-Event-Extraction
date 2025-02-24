from __future__ import annotations

import argparse

from arcee.config.experiment import ExperimentConfig
from arcee.eval.evaluator import Evaluator


def main(config: ExperimentConfig):
    # Initialise evaluator
    evaluator = Evaluator(
        **config.data_kwargs.to_dict(),
        **config.model_kwargs.to_dict(),
        **config.prompt_kwargs.to_dict(),
    )

    # Evaluate
    # TODO: don't return outputs, don't print - log to wandb instead
    outputs = evaluator.evaluate()
    print(outputs)


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser(description="Get arguments for evaluating output")
    parser.add_argument(
        "--config_path",
        required=True,
        help="String containing the local path to the experiment config",
    )
    args = parser.parse_args()

    # Init experiment config
    config = ExperimentConfig.from_yaml(args.config_path)

    # Run
    main(config)
