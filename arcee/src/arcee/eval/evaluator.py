from __future__ import annotations

import json
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

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
    Class for evaluating an event extraction model.
    """

    def __init__(
        self,
        model_id: str,
        model_type: str,
        output_path: str,
        data_path: str,
        # TODO: add check that the prompt is doing the same as arg extraction value
        argument_extraction: bool,
        prompt_cls: str,
        schema_path: str,
        batch_size: int,
        shuffle: bool | None,
        model_init_kwargs: dict,
        prompt_init_kwargs: dict,
    ):
        """TODO: to be filled in

        Args:
            model_id (str): _description_
            model_type (str): _description_
            output_path (str): _description_
            data_path (str): _description_
            argument_extraction (bool): _description_
            prompt_cls (str): _description_
            schema_path (str): _description_
            batch_size (int): _description_
            shuffle (bool | None): _description_
            model_init_kwargs (dict): _description_
            prompt_init_kwargs (dict): _description_

        Raises:
            FileExistsError: _description_
            ValueError: _description_
        """
        # Check output path doesn't already exist
        if os.path.exists(output_path):
            err_msg = f"Output path {output_path} already exists."
            raise FileExistsError(err_msg)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_path = output_path

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

        # Load data schema and create instance of prompt constructor
        schema = load_json(schema_path)
        prompt_constructor = getattr(prompts, prompt_cls)(schema, **prompt_init_kwargs)

        # Shuffle defaults to False
        if shuffle is None:
            shuffle = False

        # Instantiate data and dataloader
        self.data = EEDataset.from_file(
            data_path=data_path,
            prompt_constuctor=prompt_constructor,
            argument_extraction=argument_extraction,
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

        # TODO: total_preds, correct_preds, total_golds = 0, 0, 0
        for batch in tqdm(iter(self.dataloader)):
            batch_outputs = self.model.predict(batch)
            outputs.extend(batch_outputs)
            with open(self.output_path, "a") as f:
                for output in batch_outputs:
                    f.write(json.dumps(output) + "\n")
            # TODO:
            # - Fetch correct labels from gold data - will want to use
            # self.argument_extraction to determine whether to extract gold
            # triggers or gold arguments
            # - Save correct predictions, total predicts and total golds
            # - Save to wandb?

        # Return
        return outputs
