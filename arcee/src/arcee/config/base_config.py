from __future__ import annotations

from abc import ABC, abstractmethod

import yaml


class BaseConfig(ABC):
    """Base class for config files."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @classmethod
    def from_dict(cls, dict: dict) -> BaseConfig:
        """Create a Config from a dict.

        Args:
            config: Dict containing arguments with which to initialise the Config.

        Returns:
            Config object.
        """
        return cls(**dict)

    @classmethod
    def from_yaml(cls, path: str) -> BaseConfig:
        """Create a Config from a yaml file.

        Args:
            path: Path to yaml file from which a config dict can be read.

        Returns:
            Config object.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        raise NotImplementedError
