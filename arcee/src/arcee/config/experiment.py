from __future__ import annotations

from arcee.config.base_config import BaseConfig
from arcee.config.data_arguments import DataArguments
from arcee.config.model_arguments import ModelArguments
from arcee.config.prompt_arguments import PromptArguments


class ExperimentConfig(BaseConfig):
    """
    Config class for a single experiment.

    Attributes:
        data_kwargs: An instance of arcee.config.data_arguments.DataArguments
        model_kwargs: An instance of arcee.config.model_arguments.ModelArguments
        prompt_kwargs: An instance of arcee.config.prompt_arguments.PromptArguments
    """

    def __init__(
        self,
        data_kwargs: dict,
        model_kwargs: dict,
        prompt_kwargs: dict,
    ) -> None:
        super().__init__()
        self.data_kwargs = DataArguments.from_dict(data_kwargs)
        self.model_kwargs = ModelArguments.from_dict(model_kwargs)
        self.prompt_kwargs = PromptArguments.from_dict(prompt_kwargs)

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "data_kwargs": self.data_kwargs.to_dict(),
            "model_kwargs": self.model_kwargs.to_dict(),
            "prompt_kwargs": self.prompt_kwargs.to_dict(),
        }
