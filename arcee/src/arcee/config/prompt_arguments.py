from __future__ import annotations

from arcee.config.base_config import BaseConfig


class PromptArguments(BaseConfig):
    def __init__(
        self,
        prompt_cls: str,
    ) -> None:
        super().__init__()
        self.prompt_cls = prompt_cls

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "prompt_cls": self.prompt_cls,
        }
