from __future__ import annotations

from torch.utils.data import Dataset

from arcee.dataset.load import load_data
from arcee.prompts.prompts import EEPrompt


# Please ignore docstrings for now - I'm ripping this off selective forgetting
class EEDataset(Dataset):
    """
    Dataset class used for conventionally fine-tuning models on a whole question-answer
    dataset. __getitem__ returns a single tokenized & formatted question--answer pair
    (conctenated to a single string).
    """

    def __init__(
        self,
        data: list[dict],
        prompt_constuctor: EEPrompt,
        event_schema: dict,
        **kwargs: str | int,
    ):
        """
        Dataset which returns a tokenized version of the input given a tokenizer and
        question-answer formatting function.

        Args:
            data: torch Dataset containing data for the dataset
            tokenizer : Used to tokenize the input
            qa_formatter : QAFormatter instance used to format input before passing it
                to the model
        """
        super().__init__()
        self.data = data
        self.prompt_constuctor = prompt_constuctor
        self.event_schema = event_schema
        self.kwargs = kwargs

    @classmethod
    def from_file(
        cls,
        data_path: str,
        prompt_constuctor: EEPrompt,
        event_schema: dict,
        argument_extraction: bool = False,
        **kwargs: str | int,
    ) -> EEDataset:
        data = load_data(data_path, record_per_event=argument_extraction)
        # TODO: remove temporary shorterning of dataset
        data = data[0:16]
        return cls(data, prompt_constuctor, event_schema, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        prompt, output_format = self.prompt_constuctor.get_prompt(
            tokens=self.data[idx]["tokens"],
            event_mentions=self.data[idx]["event_mentions"],
            schema=self.event_schema,
            **self.kwargs,
        )
        return {
            "role": "user",
            "content": prompt,
        }, output_format
