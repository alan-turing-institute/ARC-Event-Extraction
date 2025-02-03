from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import ollama


class EEBaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, test_data_path: str) -> None:
        pass


class EEBaseICLModel(EEBaseModel):
    """
    To do: class for in-context learning (ICL) only, i.e., not fine-tuning
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, test_data_path: str, prompt_template: str = "") -> None:
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

    def _get_response(self, messages: list[dict[Any, Any]]) -> str | None:
        """
        Get the response from the Ollama model.

        This will be in a JSON format.

        Args:
            messages (list[dict[Any, Any]]): The list of messages to send to the model.
                They are in the format {"role": "user", "content": "message"}.

        Returns:
            str | None: This is the response from the model. It will only include the
                content of the message. Should be in JSON format.
        """
        return self.ollama_client.chat(
            self.model,
            messages=messages,
            # format="json",
            options=self.options,
            keep_alive=self.keep_alive,
        )["message"]["content"]

    def predict(self, test_data_path: str, prompt_template: str = "") -> None:
        path = test_data_path
        message = prompt_template
        response = self._get_response([{"role": "user", "content": message}])
        print(path, response)
