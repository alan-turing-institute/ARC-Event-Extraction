from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import ollama

from arcee.prompts.prompts import EEPrompt
from arcee.utils.data import load_data, load_pattern


class EEBaseModel(ABC):
    """
    Base class for event extraction models.
    The classes that will inherit from this will be wrapper classes for
    HuggingFace or Ollama models.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(
        self,
        data_file_path: str,
        pattern_file_path: str,
        prompt_obj: EEPrompt,
    ) -> None:
        """
        Makes predictions for some data, whether it is doing event detection (ED),
        event argument extraction (EAE) or end-to-end (E2E) will be determined
        by the logic held in the prompt.

        TO DO: Determine the required output format, currently set to None
        but will want to return the predictions in some form...
        """


class EEMockModel(EEBaseModel):
    """
    Only used in tests to avoid loading a Ollama or HF model while running tests
    """

    def __init__(self) -> None:
        super().__init__()

    def predict(
        self,
        data_file_path: str,
        pattern_file_path: str,
        prompt_obj: EEPrompt,
    ) -> None:
        pass


class EEOllamaModel(EEBaseModel):
    """
    Wrapper class to conduct event extraction tasks using the Ollama framework
    """

    def __init__(
        self,
        model: str,
        options: dict | None = None,
        keep_alive: int = 300,
        host_port: int = 11434,
    ) -> None:
        super().__init__()
        """
        Parameters
        ----------
        model: str
            The name of an Ollama model,
        """
        super().__init__()
        if options is None:
            options = {"temperature": 0.0}
        self.options = options

        self.keep_alive = keep_alive

        if model not in [mod["model"] for mod in ollama.list()["models"]]:
            print("Model not found. Trying to download...")
            ollama.pull(model)

        self.model = model

        self.ollama_client = ollama.Client(host=f"localhost:{host_port}")

    def _get_response(
        self,
        messages: list[dict[Any, Any]],
        response_format: str = "json",
    ) -> str | None:
        """
        Get the response from the Ollama model.

        Parameters:
        -----------
        messages : list[dict[Any, Any]])
            The list of messages to send to the model. They are in the format
            {"role": "user", "content": "message"}.
        format : str
            Expected 'json' if JSON format is to be used

        Returns:
            str | None: This is the response from the model. It will only include the
                content of the message. Should be in JSON format if the "json" format
                is set.
        """
        return self.ollama_client.chat(
            self.model,
            messages=messages,
            format=response_format,
            options=self.options,
            keep_alive=self.keep_alive,
        )["message"]["content"]

    def predict(
        self,
        data_file_path: str,
        pattern_file_path: str,
        prompt_obj: EEPrompt,
        examples_data_file_path: str | None = None,
    ) -> None:
        """
        This method makes predictions using an Ollama model, a given prompt and
        a dataset

        Parameters
        ----------
        data_file_path: str
            The file path of the data to be used for making predictions, e.g.,
            "data/RAMS/output/splits/split1/test.json"
        pattern_file_path: str
            The file path containing the schema (currently taken from the pattern files
            in TextEE) e.g., "data/RAMS/patterns/pattern.txt"
        prompt_obj: EEPrompt:
            An object that inherits from EEPrompt
        examples_data_file_path: str | None = None
            If None, then no examples will be loaded and passed to the prompt
            Otherwise, expected to be the file loaction of a dataset to be used
            as examples, e.g., "data/RAMS/output/splits/split1/train.json"

        Returns
        ------
        Currently returns None, but will want to update once we've worked out a
        suitable format.

        """
        # Load data to make predictions on
        test_data = load_data(data_file_path)

        # Load schema
        schema = load_pattern(pattern_file_path, "rams")

        # If using examples, then load the dataset from which the examples
        # can be sampled
        if examples_data_file_path is not None:
            examples_data = load_data(examples_data_file_path)

        # Temp for now: hard-coded to first ten records in dataset
        for data in test_data[:10]:
            system_content = prompt_obj.get_system_content()
            prompt, response_format = prompt_obj.get_prompt(
                data["tokens"],
                data["event_mentions"],
                schema,
                examples_data,
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            response = self._get_response(messages, response_format)

            # Temp for now: print the prompt, response and annotated solution
            print(prompt)
            print()
            print(response)
            print()
            print(data["event_mentions"][0]["arguments"])
            print()
            print("-------------------------------------------")
