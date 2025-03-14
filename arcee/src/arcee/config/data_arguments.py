from __future__ import annotations

from arcee.config.base_config import BaseConfig


class DataArguments(BaseConfig):
    """
    Config arguments for a dataset.

    Attributes:
        data_path: A path giving the location of a dataset to be loaded using
            `arcee.dataset.load.load_data`
        schema_path: A path giving the location of a dataset schema to be loaded using
            `arcee.dataset.load.load_schema`
        shuffle: A boolean denoting whether or not to shuffle examples in a dataloader.
            Defaults to `False` if set to `None`.
    """

    def __init__(
        self,
        data_path: str,
        schema_path: str,
        gold_path: str = "", # Only needed if saving to wandb
        only_events: bool = False,
        shuffle: bool | None = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.schema_path = schema_path
        self.gold_path = gold_path
        self.only_events = only_events
        self.shuffle = shuffle

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "data_path": self.data_path,
            "schema_path": self.schema_path,
            "gold_path": self.gold_path,
            "only_events": self.only_events,
            "shuffle": self.shuffle,
        }
