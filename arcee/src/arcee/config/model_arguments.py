from __future__ import annotations

from arcee.config.base_config import BaseConfig


class ModelArguments(BaseConfig):
    """
    Config arguments for a model.

    Attributes:
        model_id: A string giving the HuggingFace model ID, Ollama model ID, or a path
            from which to load a model.
        model_type: What type of model to use. Should be one of `"hf"` or `"ollama"`, or
            an error will be raised downstream. See arcee.eval.evaluator
        batch_size: Integer denoting the batch size to use when iterating over the
            dataset.
        model_init_kwargs: Dict containing optional kwargs with which to initialise
            the model. See classes in arcee.ee_models.ee_models
    """

    def __init__(
        self,
        model_id: str,
        model_type: str,
        batch_size: int,
        model_init_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.model_type = model_type
        self.batch_size = batch_size
        if model_init_kwargs is None:
            model_init_kwargs = {}
        self.model_init_kwargs = model_init_kwargs

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "model_init_kwargs": self.model_init_kwargs,
        }
