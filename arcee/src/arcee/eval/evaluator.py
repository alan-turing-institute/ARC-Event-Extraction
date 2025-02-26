from __future__ import annotations

from torch.utils.data import DataLoader

import arcee.prompts.prompts as prompts
from arcee.dataset.dataset import EEDataset
from arcee.dataset.load import load_json
from arcee.ee_models.ee_models import EEHFModel, EEOllamaModel

MODEL_CLS_MAP = {
    "hf": EEHFModel,
    "ollama": EEOllamaModel,
}


class Evaluator:
    """
    A class which manages and runs evaluations.

    Attributes:
        model_id: A string denoting the id or path of the model being used. Should be a
            hugggingface model id, an ollama model id, or a path to a model.
        model: A model whose class inherits from EEBaseModel. The model type is selected
            using the `model_type` argument, which should one one of `"hf"` or
            `"ollama"`. Additional arguments conditional on the model class can be
            passed via the `model_init_kwargs` dictionary
        data: An EEDataset. It is initialised using the `data_path` argument denoting a
            dataset to be loaded, the `prompt_cls` which is a string for a prompt
            construction class in arcee.prompts.prompts.py, and the `schema_path` path
            argument which is the path to the event schema for that dataset.
        dataloader: A torch.DataLoader, used for batching the dataset. It is initialised
            using the dataset, the `shuffle` argument` and the `batch_size` argument.
            See the torch documentation for details.
    """

    def __init__(
        self,
        model_id: str,
        model_type: str,
        data_path: str,
        prompt_cls: str,
        schema_path: str,
        batch_size: int,
        shuffle: bool | None,
        model_init_kwargs: dict,
        prompt_kwargs: dict,
        argument_extraction: bool = False,  # dataset
    ):
        # Store model ID
        # TODO: handle trained models stored locally differently? stored on wandb?
        self.model_id = model_id

        # Instantiate model
        if model_type not in ["hf", "ollama"]:
            exception = (
                "`model_type` should be one of `'hf'` or `'ollama'` `model_type` is "
                f"currently being passed as {model_type}."
            )
            raise ValueError(exception)
        model_cls = MODEL_CLS_MAP[model_type]
        self.model = model_cls(
            model_id=self.model_id,
            **model_init_kwargs,
        )

        # Get prompter and data schema
        prompt_constructor = getattr(prompts, prompt_cls)()
        schema = load_json(schema_path)

        # Shuffle defaults to False
        if shuffle is None:
            shuffle = False

        # Instantiate data and dataloader
        self.data = EEDataset.from_file(
            data_path=data_path,
            prompt_constuctor=prompt_constructor,
            event_schema=schema,
            argument_extraction=argument_extraction,
            **prompt_kwargs,
        )
        self.dataloader = DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: x,
        )

        # This will be needed when we want to know how to extract the results
        # i.e., trigger or arg extraction
        self.argument_extraction = argument_extraction
        # TODO: handle result extraction
        # self.extractor= extractor

    def evaluate(self) -> list[str]:
        """
        When this method is called, the evaluator will perform evaluation by batching
        over the dataset and passing inputs to the model.
        """
        # Get model ouputs and decode using tokenzier
        outputs = []

        for batch in iter(self.dataloader):
            outputs.extend(self.model.predict(batch))

        # Extract trigger/arguments from predicted outputs (given dataset)
        # TODO: set up EE dataset to return targets conditional on schema
        # TODO: trigger, arg, or both arg

        # Compute f1 scores
        # TODO: conditional on above

        # Log scores
        # TODO: get wandb setup

        # TODO: remove return statement once metrics are being logged (and set return
        # to None type)
        return outputs
