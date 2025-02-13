from __future__ import annotations

from arcee.config.base_config import BaseConfig


class ModelArguments(BaseConfig):
    def __init__(
        self,
        model_id: str,
        model_type: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.model_type = model_type
        self.batch_size = batch_size

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
        }
