from __future__ import annotations

from torch.utils.data import DataLoader

import arcee.prompts.prompts as prompts
from arcee.dataset.dataset import EEDataset
from arcee.dataset.load import load_pattern
from arcee.ee_models.ee_models import EEHFModel, EEOllamaModel

MODEL_CLS_MAP = {
    "hf": EEHFModel,
    "ollama": EEOllamaModel,
}


class Evaluator:
    def __init__(
        self,
        model_id: str,  # model
        model_type: str,  # model
        data_path: str,  # dataset
        prompt_constructor_cls: str,  # dataset/prompt constructor
        pattern_file_path: str,  # dataset/prompt constructor
        batch_size: int,  # loader
        shuffle: bool = False,  # loader
        model_options: dict | None = None, # model
        **kwargs: str | int,  # prompt constructor
        # extractor: BaseExtractor,
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
            options = model_options,
            # TODO: other kwargs probably needed
        )

        # Get prompter and data schema
        prompt_constructor = getattr(prompts, prompt_constructor_cls)()
        schema = load_pattern(pattern_file_path)

        # Instantiate data and dataloader
        self.data = EEDataset.from_file(
            data_path=data_path,
            prompt_constuctor=prompt_constructor,
            event_schema=schema,
            **kwargs,
        )
        self.dataloader = DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: x,
        )

        # TODO: handle result extraction
        # self.extractor= extractor

    def evaluate(self) -> list[str]:
        # Get model ouputs and decode using tokenzier
        # TODO: use dataloader, loop over batches to produce outputs
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

        # TODO: remove return statement
        return outputs
