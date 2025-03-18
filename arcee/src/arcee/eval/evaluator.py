from __future__ import annotations

import json
import os
from typing import Any

from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

import arcee.prompts.prompts as prompts
from arcee.dataset.dataset import EEDataset
from arcee.dataset.load import load_json
from arcee.ee_models.ee_models import EEHFModel, EEOllamaModel
from arcee.ee_models.guide_models import (
    GuideFreeModel,
    GuideJSONModel,
    GuideSchemaModel,
)
from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)

MODEL_CLS_MAP = {
    "hf": EEHFModel,
    "ollama": EEOllamaModel,
    "guide_free": GuideFreeModel,
    "guide_json": GuideJSONModel,
    "guide_schema": GuideSchemaModel,
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
        only_events: bool,
        prompt_cls: str,
        schema_path: str,
        gold_path: str,
        batch_size: int,
        shuffle: bool | None,
        wandb_run: Run | None,
        model_init_kwargs: dict,
        prompt_init_kwargs: dict,
    ):
        """
        Class to evaluate an event extraction model.

        Args:
            model_id (str): A string ID for the model, e.g., "Llama3.1:8b"
            model_type (str): Must match one of the keys in MODEL_CLS_MAP, i.e., "hf",
                "ollama", "guide_free", "guide_json", "guide_schema"
            output_path (str): The path to save the model output
            data_path (str): The path to the data to evaluate
            only_events (bool): Is True if evaluation is only to be carried out on
                records with events
            prompt_cls (str): The name of the prompt class to use
            schema_path (str): The path to the data schema
            gold_path (str): The path to the gold labelled data
            batch_size (int): The batch size to use for evaluation
            shuffle (bool | None): Is True if the data should be shuffled
            model_init_kwargs (dict): Model initialisation kwargs
            prompt_init_kwargs (dict): Prompt initialisation kwargs

        Raises:
            FileExistsError: Thrown if the output file already exists (to avoid
                overwriting)
            ValueError: Thrown if the model type is not in MODEL_CLS_MAP
        """
        self.wandb_run = wandb_run

        # Check output path doesn't already exist
        if os.path.exists(output_path):
            err_msg = f"Output path {output_path} already exists."
            raise FileExistsError(err_msg)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_path = output_path

        # Store model ID
        self.model_id = model_id

        # Instantiate model
        if model_type not in MODEL_CLS_MAP:
            exception = (
                f"`model_type` should be one of {list(MODEL_CLS_MAP.keys())} "
                "`model_type` is "
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
        self.argument_extraction = prompt_constructor.get_is_arg_extraction()

        # Load the gold-labelled data
        self.record_id = "evt_id" if self.argument_extraction else "wnd_id"
        self.gold_data = load_events(gold_path, self.record_id)
        self.eval_func: Any = (
            stats_on_argument_detection
            if self.argument_extraction
            else stats_on_event_detection
        )

        # This will be needed when we want to know how to extract the results
        # i.e., do we only want to compare with records that have events
        self.only_events = only_events or self.argument_extraction

        # Shuffle defaults to False
        if shuffle is None:
            shuffle = False

        # Instantiate data and dataloader
        self.data = EEDataset.from_file(
            data_path=data_path,
            prompt_constuctor=prompt_constructor,
            only_events=self.only_events,
        )
        self.dataloader = DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: x,
        )

    def evaluate(self) -> list[str]:
        """
        When this method is called, the evaluator will perform evaluation by batching
        over the dataset and passing inputs to the model.
        """

        # Get model ouputs and decode using tokenzier
        outputs = []

        for batch in tqdm(iter(self.dataloader)):
            batch_outputs = self.model.predict(batch)
            outputs.extend(batch_outputs)
            with open(self.output_path, "a") as f:
                for output in batch_outputs:
                    f.write(json.dumps(output) + "\n")
            if self.wandb_run and self.gold_data:
                outputs_to_compare = load_events(self.output_path, self.record_id)
                if outputs_to_compare:
                    stats = self.eval_func(self.gold_data, outputs_to_compare)
                    self.wandb_run.log(stats)

        # Return
        return outputs
