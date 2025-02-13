from __future__ import annotations

import argparse

from arcee.config.experiment import ExperimentConfig
from arcee.eval.evaluator import Evaluator


def main(config: ExperimentConfig):
    # Initialise evaluator
    evaluator = Evaluator(
        model_id=config.model_kwargs.model_id,
        model_type=config.model_kwargs.model_type,
        # dataset_name=config.data_kwargs.dataset_name,
        data_path=config.data_kwargs.data_path,
        prompt_constructor_cls=config.prompt_kwargs.prompt_cls,
        pattern_file_path=config.data_kwargs.schema_path,
        batch_size=config.model_kwargs.batch_size,
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
