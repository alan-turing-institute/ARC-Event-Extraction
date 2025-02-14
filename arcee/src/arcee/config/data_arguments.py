from __future__ import annotations

from arcee.config.base_config import BaseConfig


class DataArguments(BaseConfig):
    def __init__(
        self,
        data_path: str,
        schema_path: str,
        shuffle: bool | None = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.schema_path = schema_path
        self.shuffle = shuffle

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "data_path": self.data_path,
            "schema_path": self.schema_path,
            "shuffle": self.shuffle,
        }
