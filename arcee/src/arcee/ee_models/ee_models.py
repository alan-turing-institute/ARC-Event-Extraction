from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import ollama

from arcee.prompts.prompts import EAEPrompt
from arcee.utils.data import load_data, load_pattern


class EEBaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict_ed(self, data_file_path: str) -> None:
        pass

    @abstractmethod
    def predict_eae(
        self,
        data_file_path: str,
        pattern_file_path: str,
        prompt_obj: EAEPrompt,
    ) -> None:
        pass


class EEBaseICLModel(EEBaseModel):
    """
    To do: base class for in-context learning (ICL) only, i.e., not fine-tuning
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict_ed(self, test_data_path: str, prompt_template: str = "") -> None:
        pass

    @abstractmethod
    def predict_eae(
        self,
        data_file_path: str,
        pattern_file_path: str,
        prompt_obj: EAEPrompt,
        # use_examples: bool = False,
        # examples_data_file_path: str | None = None,
    ) -> None:
        pass


class EEOllamaModel(EEBaseICLModel):
    """
    To do: class for using Ollama to do in-context learning for EE
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
        To do
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

    def predict_ed(self, test_data_path: str, prompt_template: str = "") -> None:
        pass

    def predict_eae(
        self,
        data_file_path: str,
        pattern_file_path: str,
        prompt_obj: EAEPrompt,
        # use_examples: bool = False,
        # examples_data_file_path: str | None = None,
    ) -> None:
        test_data = load_data(data_file_path)
        pattern = load_pattern(pattern_file_path, "rams")
        # examples_data = []
        # if use_examples:
        #     examples_data = load_data(examples_data_file_path)

        for data in test_data[:10]:
            system_content = prompt_obj.get_system_content()
            prompt, response_format = prompt_obj.get_prompt(
                data["tokens"], data["event_mentions"], pattern
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            response = self._get_response(messages, response_format)
            print(prompt)
            print()
            print(response)
            print()
            print(data["event_mentions"][0]["arguments"])
            print()
            print("-------------------------------------------")
