from __future__ import annotations

import json
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
        input: list[tuple[dict[str, str], str | dict, dict]],
    ) -> list[dict]:
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
            # Set temperature to zero (less creative)
            # Ollama's context window length (option `num_ctx`) defaults to 2048, which
            # is likely to be too short for many prompts, so setting this to 8192.
            options = {"temperature": 0.0, "num_ctx": 8192}
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
        response_format: str | dict = "json",
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
        self, input: list[tuple[dict[str, str], str | dict, dict]]
    ) -> list[dict]:
        """
        This method makes predictions using an Ollama model, a given prompt and
        a dataset

        Parameters
        ----------
        input: list[tuple[dict[str, str], str]]
            A list of messages to be passed to the Ollama client chat

        Returns
        -------
        list[dict[str, str | list | dict]]
            A list of dictionaries, with the record IDs and the output from the model
            If the output is not in JSON format, then it will be logged with a key of
            'JSON_ERROR' and the value of the response.

        """
        outputs = []
        for record in input:
            messages = [record[0]]
            output_format = record[1]
            # Create an output dictionary that consists of the record IDs
            ids = record[2]
            response = self._get_response(messages, output_format)
            # Append the response to the output dict
            try:
                output = json.loads(response)
            except json.JSONDecodeError:
                output = {"JSON_ERROR": response}

            outputs.append({**ids, "output": output})

        return outputs


class EEMockOllamaModel(EEOllamaModel):
    """
    Only used in tests to avoid loading a Ollama or HF model while running tests
    """

    def __init__(self) -> None:
        pass

    def _get_response(
        self,
        messages: list[dict[str, str]],
        response_format: str | dict = "json",
    ) -> str:
        # To prevent the linter throwing a wobbly, do something with the args
        del messages
        del response_format
        return '{"object": "mock object"}'

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict]]
    ) -> list[dict]:
        return super().predict(input)


class EEHFModel(EEBaseModel):
    """
    Wrapper class for performing event extraction using HuggingFace transformers.

    Attributes:
        model: An instance of a `transformers.AutoModelForCausalLM`. Should be an
            instruction-tuned model whose tokenizer has a chat template.
        device: A string denoting what device to place the model on, defaults to `cpu`.
            Pass `device_map="auto"` to init to automatically place on any available
            GPU.
        tokenizer: An instance of a `transfomers.AutoTokenizer`, corresponding to the
            model. Should contain a chat template.
        terminators: A list of strings, each of which is a terminating token for
            generation.
        options: Currently ignored, kept in to match base class signature.
    """

    def __init__(
        self,
        model_id: str,
        tokenizer_path: str | None = None,
        terminators: str | list[str] | None = None,
        device_map: str = "cpu",
        options: dict | None = None,
    ) -> None:
        super().__init__()
        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Setup special tokens and chat template
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.chat_template is None:
            exception_text = (
                "EEHFModel is designed only to use instruction-tuned models whose "
                "tokenizers contain a chat template. Try another model instead."
            )
            raise NotImplementedError(exception_text)
        self.tokenizer = tokenizer

        # Setup terminators
        if terminators is None:
            terminators = []
        if type(terminators) is str:
            terminators = [terminators]
        self.terminators = [
            tokenizer.eos_token_id,
            *terminators,
        ]

        # Load model
        self.device = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
        )

        # Options
        self.options = options

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict]]
    ) -> list[dict]:
        """
        Method for making predictions with a transformers model. Inputs should come
        from an `arcee.dataset.dataset.EEDataset`.

        Args:
            input: A list of dictionaries of format {"role: role, "content": content}.
        """
        # Process input into chat template
        text = self.tokenizer.apply_chat_template(
            input, add_generation_prompt=True, tokenize=False
        )
        tokenized_text = self.tokenizer(text, padding="longest", return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in tokenized_text.items()}

        # Generate outputs
        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # Decode
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
