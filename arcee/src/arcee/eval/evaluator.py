from __future__ import annotations

import json
import os
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import arcee.prompts.prompts as prompts
from arcee.dataset.dataset import EEDataset
from arcee.dataset.load import load_json
from arcee.ee_models.ee_models import EEHFModel, EEOllamaModel
from arcee.ee_models.guide_models import (
    GuideFreeModel,
    GuideJSONModel,
    GuideSchemaModel,
)
from arcee.stats.gold_compare import load_events, stats_on_argument_detection, stats_on_event_detection

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
        wandb_run: wandb.Run | None,
        model_init_kwargs: dict,
        prompt_init_kwargs: dict,
    ):
        """TODO: to be filled in

        Args:
            model_id (str): _description_
            model_type (str): _description_
            output_path (str): _description_
            data_path (str): _description_
            only_events (bool): _description_
            prompt_cls (str): _description_
            schema_path (str): _description_
            gold_path (str): 
                The path to the gold labelled data
            batch_size (int): _description_
            shuffle (bool | None): _description_
            model_init_kwargs (dict): _description_
            prompt_init_kwargs (dict): _description_

        Raises:
            FileExistsError: _description_
            ValueError: _description_
        """
        self.wandb_run = wandb_run

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
        record_id = "evt_id" if self.argument_extraction else "wnd_id"
        self.gold_data = load_events(gold_path, record_id)
        self.eval_func = stats_on_argument_detection if self.argument_extraction else stats_on_event_detection

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

        # TODO: total_preds, correct_preds, total_golds = 0, 0, 0
        for batch in tqdm(iter(self.dataloader)):
            batch_outputs = self.model.predict(batch)
            outputs.extend(batch_outputs)
            with open(self.output_path, "a") as f:
                for output in batch_outputs:
                    f.write(json.dumps(output) + "\n")
            if self.wandb_run:
                stats = self.eval_func(self.gold_data, outputs, semi=False)
                self.wandb_run.log(stats)

        # Return
        return outputs
