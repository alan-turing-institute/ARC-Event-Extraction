from __future__ import annotations

from abc import ABC, abstractmethod

import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        input: list[dict[str, str]],
    ) -> list[str]:
        """
        Makes predictions for some data, whether it is doing event detection (ED),
        event argument extraction (EAE) or end-to-end (E2E) will be determined
        by the logic held in the prompt.
        """


class EEOllamaModel(EEBaseModel):
    """
    Wrapper class to conduct event extraction tasks using the Ollama framework
    """

    def __init__(
        self,
        model_id: str,
        options: dict | None = None,
        keep_alive: int = 300,
        host_port: int = 11434,
    ) -> None:
        super().__init__()
        """
        Parameters
        ----------
        model: str
            The name of an Ollama model, will try to download if it doesn't already
            exist
        options: dict | None
            These are used when the chat function is called - see Ollama docs for
            possible values here:
            https://github.com/ollama/ollama/blob/main/docs/api.md
            If nothing is passed, then the temperature is set to zero
        keep_alive: int
            controls how long the model will stay loaded into memory following the
            request (default is 5m)
        host_port: int
            Needed when running on HPC to make sure the Ollama models don't interfere
            with one another

        """
        super().__init__()
        if options is None:
            # TODO: Copied this functionality over from WikiGame, do we want it here?
            # Might want to see the difference in performance when changing the
            # temperature, I have a feeling a lower value might be better to stop it
            # being too creative when doing event extraction, but not 100% sure
            options = {"temperature": 0.0}
        self.options = options

        self.keep_alive = keep_alive

        if model_id not in [mod["model"] for mod in ollama.list()["models"]]:
            print("Model not found. Trying to download...")
            ollama.pull(model_id)

        self.model = model_id

        self.ollama_client = ollama.Client(host=f"localhost:{host_port}")

    def _get_response(
        self,
        messages: list[dict[str, str]],
        response_format: str = "json",
    ) -> str:
        """
        Get the response from the Ollama model.

        Parameters:
        -----------
        messages : list[dict[Any, Any]])
            The list of messages to send to the model. They are in the format
            {"role": "user", "content": "message"}.
        response_format : str
            Expected 'json' if JSON format is to be used
            If blank then natural language is returned
            Might want to add a check that nothing else is passed?

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
        input: list[dict[str, str]],
    ) -> list[str]:
        """
        This method makes predictions using an Ollama model, a given prompt and
        a dataset

        Parameters
        ----------
        input: list[dict[str, str]]
            A list of messages to be passed to the Ollama client chat

        Returns
        -------
        list[str]
            The response from the Ollama model

        """
        responses = []
        for message in input:
            responses.append(self._get_response([message], "json"))

        return responses


class EEMockOllamaModel(EEOllamaModel):
    """
    Only used in tests to avoid loading a Ollama or HF model while running tests
    """

    def __init__(self) -> None:
        pass

    def _get_response(
        self,
        messages: list[dict[str, str]],
        response_format: str = "json",
    ) -> str:
        # To prevent the linter throwing a wobbly, do something with the args
        del messages
        del response_format
        return '{"object": "mock object"}'

    def predict(self, input: list[dict[str, str]]) -> list[str]:
        return [self._get_response(input, "json")]


class EEHFModel(EEBaseModel):
    def __init__(
        self,
        model_id: str,
        options: dict | None = None,
    ) -> None:
        super().__init__()
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # TODO: Llama only?
        ]

        # Load model
        # TODO additional kwargs
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        # Options (TODO: needed?)
        if options is None:
            options = {"temperature": 0.0}
        self.options = options

    def predict(
        self,
        input: list[dict[str, str]],
    ) -> list[str]:
        # Process input into chat template
        # TODO reconcile w/ prompt classes, data format (lost[dict[str]]) above
        text = self.tokenizer.apply_chat_template(
            input, add_generation_prompt=True, tokenize=False
        )
        tokenized_text = self.tokenizer(text, padding="longest", return_tensors="pt")

        # Generate outputs
        gen_tokens = self.model.generate(
            **tokenized_text,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # Decode
        # TODO: needs class wrapped around output?
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
