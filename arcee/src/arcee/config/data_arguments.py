from __future__ import annotations

from arcee.config.base_config import BaseConfig


class DataArguments(BaseConfig):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        schema_path: str,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.schema_path = schema_path

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "dataset_name": self.dataset_name,
            "data_path": self.data_path,
            "schema_path": self.schema_path,
        }
