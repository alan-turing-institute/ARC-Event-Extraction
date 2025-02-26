from __future__ import annotations

from arcee.config.base_config import BaseConfig


class PromptArguments(BaseConfig):
    """
    Config arguments for prompt construction.

    Attributes:
        prompt_cls: The prompt class to use. String should match a class inside
            arcee.prompts.prompts.
    """

    def __init__(
        self,
        prompt_cls: str,
        prompt_const_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.prompt_cls = prompt_cls
        if prompt_const_kwargs is None:
            prompt_const_kwargs = {}
        self.prompt_const_kwargs = prompt_const_kwargs

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "prompt_cls": self.prompt_cls,
            "prompt_const_kwargs": self.prompt_const_kwargs,
        }
