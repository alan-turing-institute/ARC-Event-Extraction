from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import ollama
from openai import AzureOpenAI, BadRequestError
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
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
    ) -> list[dict]:
        """
        Makes predictions for some data, whether it is doing event detection (ED),
        event argument extraction (EAE) or end-to-end (E2E) will be determined
        by the logic held in the prompt.

        Args:
            input: list[tuple[str, str | dict, dict, str]]
                A list of tuples. Each tuple will contain:
                - dict[str,str]: The prompt stored in a dict to be used in the task.
                - str | dict: The output format, either 'json' or a JSON schema.
                - dict: A dictionary of IDs to be attached to the LLM's response
                that identify the record.
                - str: The input text

        Returns:
            list[dict[str, str | list | dict]]
                A list of dictionaries, with the record IDs and the output from the
                model. If the output is not in JSON format, then it will be logged with
                a key of 'JSON_ERROR' and the value of the response.
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
        output_format: str = "json_schema",
    ) -> None:
        super().__init__()
        """
        Args:
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
        output_format: str
            The output format required from the model. If blank then natural language
            is returned. If "json" is passed then the response will be in JSON format,
            and if "json_schema" is passed then the JSON schema from the prompt will
            be used to define the output format.

        Raises:
        ValueError: If the output_format is not one of None, 'json', or 'json_schema'

        """
        super().__init__()
        # Checks that the output format is either None, 'json' or 'json_schema'
        if output_format and output_format not in ["json", "json_schema"]:
            err_msg = "output_format must be one of '', 'json', or 'json_schema'."
            raise ValueError(err_msg)
        self.output_format = output_format

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
        response_format: str | dict,
    ) -> str:
        """
        Get the response from the Ollama model.

        Args:
        messages : list[dict[Any, Any]])
            The list of messages to send to the model. They are in the format
            {"role": "user", "content": "message"}.
        response_format : str
            If an empty string then natural language is returned
            Expected 'json' if JSON format is to be used
            If a dictionary, then this is the JSON schema to be used

        Returns:
            str | None: This is the response from the model. It will only include the
                content of the message. Should be in JSON format if the "json" format
                is set.
        """
        try:
            return self.ollama_client.chat(
                self.model,
                messages=messages,
                format=response_format,
                options=self.options,
                keep_alive=self.keep_alive,
            )["message"]["content"]
        except Exception:
            return "Error: Bad Request"

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
    ) -> list[dict]:
        """
        This method makes predictions using an Ollama model, a given prompt and
        a dataset

        Args:
        input: list[tuple[str, str | dict, dict, str]]
            A list of tuples. Each tuple will contain:
            - dict[str,str]: The prompt stored in a dict to be used in the task.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
            that identify the record.
            - str: The input text

        Returns:
        list[dict[str, str | list | dict]]
            A list of dictionaries, with the record IDs and the output from the model
            If the output is not in JSON format, then it will be logged with a key of
            'JSON_ERROR' and the value of the response.

        """
        outputs = []
        for record in input:
            messages = [record[0]]
            output_format = (
                record[1] if self.output_format == "json_schema" else self.output_format
            )
            # Create an output dictionary that consists of the record IDs
            ids = record[2]
            response = self._get_response(messages, output_format)
            # If not constrained to json or a json schema then the repsponse may
            # start with the string "```json" and end with "```" - these need to be
            # removed before it can be loaded as JSON
            formatted_response = response.replace("```json", "").replace("```", "")
            # Append the response to the output dict
            try:
                output = json.loads(formatted_response)
            except json.JSONDecodeError:
                output = {"JSON_ERROR": formatted_response}

            outputs.append({**ids, "output": output})

        return outputs


class EEMockOllamaModel(EEOllamaModel):
    """
    Only used in tests to avoid loading an Ollama model while running tests
    """

    def __init__(self, output_format: str) -> None:
        if output_format and output_format not in ["json", "json_schema"]:
            err_msg = "output_format must be one of '', 'json', or 'json_schema'."
            raise ValueError(err_msg)
        self.output_format = output_format

    def _get_response(
        self,
        messages: list[dict[str, str]],
        response_format: str | dict,
    ) -> str:
        # To prevent the linter throwing a wobbly, do something with the message arg
        del messages
        return (
            '{"response": "mock object with output format '
            + str(response_format)
            + '"}'
        )

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
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
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
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


class EEOpenAIModel(EEBaseModel):
    """
    Wrapper class for performing event extraction using OpenAI's GPT models.

    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        top_p: float = 0.1,
        seed: int = 42,
        output_format: str = "json_schema",
    ) -> None:
        super().__init__()
        # Checks that the output format is either None, 'json' or 'json_schema'
        if output_format and output_format not in ["json", "json_schema"]:
            err_msg = "output_format must be one of '', 'json', or 'json_schema'."
            raise ValueError(err_msg)

        self.model_name = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.output_format = output_format

        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def _get_response(
        self, messages: list[dict[str, str]], response_format: str | dict | None
    ) -> str:
        """
        Get the response from the OpenAI model.

        Args:
        messages : list[dict[Any, Any]])
            The list of messages to send to the model. They are in the format
            {"role": "user", "content": "message"}.
        response_format : str
            If an empty string then natural language is returned
            Expected 'json' if JSON format is to be used
            If a dictionary, then this is the JSON schema to be used

        Returns:
            str | None: This is the response from the model. It will only include the
                content of the message. Should be in JSON format if the "json" format
                is set.
        """

        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
                response_format=response_format,
            )

            return completion.choices[0].message.content
        except BadRequestError:
            return "Error: Bad Request"

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
    ) -> list[dict]:
        """
        This method makes predictions using an OpenAI model, a given prompt and
        a dataset
        TODO: JSON and JSON schema output formats are not yet implemented for ED models

        Args:
        input: list[tuple[str, str | dict, dict, str]]
            A list of tuples. Each tuple will contain:
            - dict[str,str]: The prompt stored in a dict to be used in the task.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
            that identify the record.
            - str: The input text

        Returns:
        list[dict[str, str | list | dict]]
            A list of dictionaries, with the record IDs and the output from the model
            If the output is not in JSON format, then it will be logged with a key of
            'JSON_ERROR' and the value of the response.

        """
        outputs = []
        for record in input:
            messages = [record[0]]
            output_format: dict[str, object] | None
            if self.output_format == "json":
                output_format = {"type": "json_object"}
            elif self.output_format == "json_schema" and isinstance(record[1], dict):
                updated_schema = record[1].copy()
                updated_schema["additionalProperties"] = False
                output_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "strict": True,
                        "name": "output",
                        "schema": updated_schema,
                    },
                }
            else:
                output_format = None
            # Create an output dictionary that consists of the record IDs
            ids = record[2]
            response = self._get_response(messages, output_format)
            # If not constrained to json or a json schema then the repsponse may
            # start with the string "```json" and end with "```" - these need to be
            # removed before it can be loaded as JSON
            if response:
                formatted_response = response.replace("```json", "").replace("```", "")
            else:
                formatted_response = "No response"
            # Append the response to the output dict
            try:
                output = json.loads(formatted_response)
            except json.JSONDecodeError:
                output = {"JSON_ERROR": formatted_response}

            outputs.append({**ids, "output": output})

        return outputs
