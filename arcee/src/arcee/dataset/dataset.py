from __future__ import annotations

from torch.utils.data import Dataset

from arcee.dataset.load import load_data
from arcee.prompts.prompts import EEPrompt


class EEDataset(Dataset):
    """
    Dataset class used for iterating over event extraction data while applying a valid
    prompt. Inherits from torch Dataset to enable use of the torch DataLoader

    Attributes:
        data (list[dict]): A list of dictionaries where each dictionary contains one
            piece of event extraction data, loaded by
            `arcee.dataset.load.load_dataset`
        prompt_constuctor (EEPrompt): An instance of a class inheriting from
            `arcee.prompts.prompts.EEPrompt`
    """

    def __init__(
        self,
        data: list[dict],
        prompt_constuctor: EEPrompt,
    ):
        """
        Constructor for the EEDataset class

        Args:
            data (list[dict]): A list of dictionaries where each dictionary contains one
                piece of event extraction data, loaded by
                `arcee.dataset.load.load_dataset`
            prompt_constuctor (EEPrompt): An instance of a class inheriting from
            `arcee.prompts.prompts.EEPrompt`
        """
        super().__init__()
        self.data = data
        self.prompt_constuctor = prompt_constuctor

    @classmethod
    def from_file(
        cls,
        data_path: str,
        prompt_constuctor: EEPrompt,
        only_events: bool = False,
    ) -> EEDataset:
        """
        A function for instantiating an EEDataset from a .jsonlines file.

        Args:
        data_path: str
            String denoting a path to a .jsolines file with which to instantiate the
            dataset by loading using `arcee.dataset.load.load_dataset`
        prompt_constructor: EEPrompt
            An instance of a class inheriting from `arcee.prompts.prompts.EEPrompt`
        only_events: bool
            If True then one record per event type will be loaded
        """
        data = load_data(data_path, record_per_event=only_events)
        return cls(data, prompt_constuctor)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str | dict, dict, str]:
        """
        Gets the `idx`th item of the dataset, applies the prompt to it, then returns
        in a dictionary ready for passing to a model.

        Args:
            idx: An integer denoting the index of the item of the dataset to return.
        """
        prompt, output_format, record_ids, input_text = (
            self.prompt_constuctor.get_prompt(
                data=self.data[idx],
            )
        )
        return (
            {
                "role": "user",
                "content": prompt,
            },
            output_format,
            record_ids,
            input_text,
        )
