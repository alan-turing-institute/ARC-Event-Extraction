from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

from arcee.dataset.load import load_data
from arcee.dataset.schemas import (
    data_to_ed_schema,
    get_arg_roles,
    get_ed_json_schema,
    pattern_to_json_schema,
)


class EEPrompt(ABC):
    """
    Abstract class for prompt construction. This will define common functionality
    that child prompt classes will inherit.
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_system_content(self) -> str:
        """
        Returns the system content message to be used.

        This method provides a default system content message, which can be
        overridden by a child class.

        Returns:
            str: The system content message.
        """
        return (
            "You are an assistant that helps conducts event extraction "
            "from input text.\n"
        )

    def _confirm_ed_data_format(self, data: dict[str, str | int | list]) -> str:
        """
        Validates the format of the input data and ensures the presence of the 'text'
        key.

        This method checks if the 'text' key exists in the input data and verifies that
        its value is a string. If the validation fails, appropriate exceptions are
        raised. If the validation passes, the value of the 'text' key is returned.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the input data.

        Returns:
            str: The value of the 'text' key from the input data.

        Raises:
            KeyError: If the 'text' key is missing from the data.
            TypeError: If the value of 'text' is not a string.
        """
        # Double check keys and types of data
        if "text" not in data:
            err_msg = "Key 'text' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(data["text"], str):
            err_msg = (
                "Expected 'text' to be a string, but got "
                f"{type(data['text']).__name__}."
            )
            raise TypeError(err_msg)

        return data["text"]

    def _confirm_arg_data_format(
        self, data: dict[str, str | int | list]
    ) -> tuple[dict, list]:
        """
        Validates the format of the input data for argument extraction.

        This method checks if the required keys ('event_mentions' and 'tokens') exist
        in the input data and verifies their types. If the validation fails, appropriate
        exceptions are raised. If the validation passes, then the 'event_mentions' and
        'tokens' values are returned.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the input data.

        Returns:
            tuple[dict, list]: A tuple containing:

            - dict: The first event mention from 'event_mentions'.
            - list: The list of tokens from 'tokens'.

        Raises:
            KeyError: If 'event_mentions' or 'tokens' is missing from the data.
            TypeError: If 'event_mentions' is not a list, if its first element is not
                a dictionary, or if 'tokens' is not a list.
        """
        # Double check keys and types of data
        if "event_mentions" not in data:
            err_msg = "Key 'event_mentions' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(data["event_mentions"], list):
            err_msg = (
                "Expected 'event_mentions' to be a list, but got "
                f"{type(data['event_mentions']).__name__}."
            )
            raise TypeError(err_msg)
        if not isinstance(data["event_mentions"][0], dict):
            err_msg = (
                "Expected 'event_mentions' to contain a dict, but got "
                f"{type(data['event_mentions'][0]).__name__}."
            )
            raise TypeError(err_msg)
        event_mention = data["event_mentions"][0]
        if "tokens" not in data:
            err_msg = "Key 'tokens' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(data["tokens"], list):
            err_msg = (
                "Expected 'tokens' to be a list, but got "
                f"{type(data['tokens']).__name__}."
            )
            raise TypeError(err_msg)

        return event_mention, data["tokens"]

    def _get_event_trigger(
        self, event_mention: dict[str, dict[str, int] | str]
    ) -> tuple[int, int, str]:
        """
        Extracts the trigger information from an event mention.
        The event mention is expected to be in the same format as the processed
        data from TextEE.

        Parameters
        ----------
        event_mention : dict[str, int | str]
            A dictionary containing event mention data with a "trigger" and
            "event_type" keys.

        Returns
        -------
        tuple of (int, int, str)
            A tuple containing:
                - start (int): The starting index of the event trigger.
                - end (int): The exclusive ending index of the event trigger.
                - event_type (str): The type of the event.
        """
        trigger = event_mention["trigger"]  # `trigger` has type dict[str, int]
        if not isinstance(trigger, dict) or not isinstance(
            event_mention["event_type"], str
        ):
            err_msg = "Invalid event_mention format"
            raise TypeError(err_msg)

        start = trigger["start"]
        end = trigger["end"]
        event_type = event_mention["event_type"]

        return start, end, event_type

    def _get_tagged_text(
        self, tokens: list[str], trig_start: int, trig_end: int
    ) -> str:
        """
        Inserts trigger tags into a list of tokens at the specified start and end
        indices and returns the modified text.

        Args:
            tokens (list[str]):
                A list of tokens representing a text sequence.
            trig_start (int):
                The starting index of the trigger span.
            trig_end (int):
                The exclusive ending index of the trigger span.

        Returns:
            str: The modified text with `<trigger>` and `</trigger>` tags
            surrounding the trigger span.

        Raises:
            ValueError: If `trig_start` is not less than `trig_end` or if `trig_end`
            exceeds the list bounds.
        """
        if not (0 <= trig_start < trig_end <= len(tokens)):
            err_msg = (
                f"Invalid trigger indices: trig_start={trig_start}, "
                f"trig_end={trig_end}, "
                f"must satisfy 0 <= trig_start < trig_end <= {len(tokens)}."
            )
            raise ValueError(err_msg)

        tagged_tokens = tokens.copy()
        tagged_tokens[trig_start] = "<trigger>" + tagged_tokens[trig_start]
        tagged_tokens[trig_end - 1] = tagged_tokens[trig_end - 1] + "</trigger>"

        return " ".join(tagged_tokens)

    def _get_ids(self, data: dict[str, str | int | list]) -> dict[str, str]:
        """
        Retrieves the document and window IDs to be attached to the LLM's response
        for record identification.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the input data.

        Returns:
            dict[str, str]: A dictionary containing the window ID (`wnd_id`).

        Raises:
            KeyError: If the `wnd_id` key is missing from the data.
            TypeError: If the value of `wnd_id` is not a string.
        """
        if "wnd_id" not in data:
            err_msg = "Key 'wnd_id' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(data["wnd_id"], str):
            err_msg = (
                "Expected 'wnd_id' to be a str, but got "
                f"{type(data['wnd_id']).__name__}."
            )
            raise TypeError(err_msg)
        return {"wnd_id": data["wnd_id"]}

    def _check_required_keys(self, required_keys: dict, kwargs: dict) -> None:
        """
        Validates that all required keys are present in the kwargs dictionary and have
        the correct types.

        Args:
            required_keys (dict):
                A dictionary mapping required keyword argument names to their expected
                types.
            kwargs (dict):
                A dictionary containing keyword arguments to be validated.

        Raises:
            TypeError: If a required key is missing from `kwargs`.
            TypeError: If a required key has a value of an incorrect type.
        """

        for key, expected_type in required_keys.items():
            if key not in kwargs:
                err_msg = f"Missing required keyword argument: '{key}'"
                raise TypeError(err_msg)
            if not isinstance(kwargs[key], expected_type):
                err_msg = (
                    f"Expected '{key}' to be of type {expected_type.__name__}, "
                    "but got {type(kwargs[key]).__name__}"
                )
                raise TypeError(err_msg)

    @abstractmethod
    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Abstract method for constructing the prompt.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the input data.
            schema (dict):
                A dictionary defining the schema for the task.
            **kwargs:
                Additional keyword arguments.

        Returns:
            tuple[str, str | dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
            that identify the record.
        """

    def get_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Returns a prompt, which may be overridden by child classes.

        Args:
            data (dict[str, str | int | list]):
                The input data record.
            schema (dict[str, str]):
                A schema represented as a dictionary. For event type detection, this
                dictionary contains event types and their definitions. For argument
                extraction, it contains event types and their possible argument roles.
            **kwargs:
                Additional keyword arguments.

        Returns:
            tuple[str, str | dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
            that identify the record.

        """

        prompt, output_format, ids = self._construct_prompt(data, schema, **kwargs)

        return prompt, output_format, ids


class EDPrompter1(EEPrompt):
    """
    Class to construct prompts for event trigger detection:
    - does NOT use any examples in the prompt
    - requests JSON output with a description of the schema
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_system_content(self) -> str:
        """
        This method will return the system content message to be used

        Returns:
            str: The system content message.
        """
        return (
            "You are an assistant that extracts event triggers and their associated "
            "event types from input text. Many triggers are verbs and represent a "
            "change of state or action in the text."
            "\n"
            "---\n"
        )

    def _get_event_types_defs(self, schema: dict) -> str:
        """
        Retrieves event types and their definitions from the schema and returns them as
        a string.

        Args:
            schema (dict):
                A dictionary of event types and their definitions.

        Returns:
            str: A string containing the event type definitions.
        """

        return "\n".join([key + ": " + schema[key] for key in schema])

    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the data record.
            schema (dict):
                A dictionary of the possible event types and their definitions.
            **kwargs:
                Additional keyword arguments.

        Returns:
            tuple[str, str | dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """
        # Set input text from data record
        input_text = self._confirm_ed_data_format(data)

        # For event detection the schema is the list of events with definitions
        # Concatenate event type definitions into a single string, with a line break
        # between each definition
        event_type_defs = self._get_event_types_defs(schema)

        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "These are the possible event types you can chose from:\n"
                + event_type_defs
                + "\nGiven this input text:\n```"
                + input_text
                + "```\n"
                "Do any of the event types appear in the text? "
                "Not all event types will appear in the text, there may be some "
                "that appear multiple times and there may be no event types "
                "in the text.\n"
                "If you identify one or more event type, then what is the trigger "
                "word or words in the input text?"
                "\nFormat the output strictly as JSON. The output is an array of "
                "JSON objects with exactly one key-value pair. The valid keys are "
                "the event types and the values are the trigger words from the input "
                "text."
            ),
            "json",
            self._get_ids(data),
        )


class EDPrompter2(EDPrompter1):
    """
    Class to construct prompts for event trigger detection:
    - does NOT use any examples in the prompt
    - requests JSON output with a defined JSON schema
    """

    def __init__(self) -> None:
        super().__init__()

    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the data record.
            schema (dict):
                A dictionary of the possible event types and their definitions.
            **kwargs:
                Additional keyword arguments.

        Returns:
            tuple[str, str | dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """
        # Set input text from data record
        input_text = self._confirm_ed_data_format(data)

        # For event detection the schema is the list of events with definitions
        # Concatenate event type definitions into a single string, with a line break
        # between each definition
        event_type_defs = self._get_event_types_defs(schema)

        # Define the JSON schema for the output
        json_schema = get_ed_json_schema(schema)

        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "These are the possible event types you can chose from:\n"
                + event_type_defs
                + "\nGiven this text:\n```"
                + input_text
                + "```\n"
                "Do any of the event types appear in the text? "
                "Not all event types will appear in the text, there may be some "
                "that appear multiple times and there may be no event types "
                "in the text.\n"
                "If you identify one or more event type, then what is the trigger "
                "word or words in the input text?"
                + "\nFormat the output strictly as JSON. The output is an array of "
                "JSON objects with exactly one key-value pair. The valid keys are "
                "the event types and the values are the trigger words from the input "
                "text."
            ),
            json_schema,
            self._get_ids(data),
        )


class EDPrompter3(EDPrompter1):
    """
    Class to construct prompts for event trigger detection:
    - uses RANDOM samples in the prompt the prompt
    - requests JSON output with a defined JSON schema
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_examples(
        self, example_data_path: str, num_examples: int, **kwargs: Any
    ) -> list[dict]:
        """
        Selects a random subset of example records from a dataset.

        This function loads example data from a given file, sets a random seed for
        reproducibility, and selects `num_examples` random examples without replacement.

        Args:
            example_data_path (str): Path to the file containing example records.
            num_examples (int): The number of examples to randomly select.
            **kwargs: Additional keyword arguments, must include:
                - seed (int): Random seed to ensure reproducibility of the selected
                examples.

        Returns:
            list[dict]: A list of randomly selected example records.
        """
        # Check that the seed kwarg exists
        required_kwargs = {"seed": int}
        self._check_required_keys(required_kwargs, kwargs)
        seed = kwargs["seed"]
        # Load the example data
        example_data = load_data(example_data_path)
        # Set random seed
        np.random.seed(seed)
        # Return 'num_examples' random examples from the data
        return list(np.random.choice(example_data, num_examples, replace=False))

    def _construct_examples(
        self, example_data_path: str, num_examples: int, **kwargs: Any
    ) -> str:
        """
        Constructs formatted example prompts based on retrieved event records.

        This function retrieves example records using `_get_examples`, extracts relevant
        event mentions from each record, and formats the data into structured prompt
        text for model input.

        Args:
            example_data_path (str): Path to the file containing example records.
            seed (int): Random seed to ensure reproducibility when selecting examples.
            num_examples (int): The number of examples to retrieve and format.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A formatted string containing structured example prompts.
        """
        example_records = self._get_examples(example_data_path, num_examples, **kwargs)

        expected_output = data_to_ed_schema(example_records)

        output = []
        for i, record in enumerate(example_records):
            input_text = self._confirm_ed_data_format(record)

            prompt_text = (
                "### Example " + str(i + 1) + "\n"
                "Input text: ```" + input_text + "```\n"
                "Expected Response (JSON): \n" + json.dumps(expected_output[i]) + "\n"
            )
            output.append(prompt_text)
        return "\n".join(output)

    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        This function generates a structured prompt for event detection by
        combining input text, event type definitions, and example texts. It
        also defines the expected output format using a JSON schema.

        Args:
            data (dict[str, str | int | list]):
                The input data record containing text and other metadata.
            schema (dict):
                A dictionary defining possible event types and their descriptions.
            **kwargs:
                Additional parameters for constructing the prompt, must include:
                - example_data_path (str):
                  Path to the file containing example event records.
                - num_examples (int):
                  The number of example records to include in the prompt.
                Might also include:
                - example_kwargs (dict):
                  Additional keyword arguments for selecting examples.

        Returns:
            tuple[str, str | dict, dict]: A tuple containing:
                - str: The constructed prompt with event type definitions,
                  examples, and input text.
                - str | dict: The JSON schema defining the expected output format.
                - dict: A dictionary containing extracted IDs from the input data.
        """
        # Check that we have the expected kwargs
        required_keys = {"example_data_path": str, "num_examples": int}
        self._check_required_keys(required_keys, kwargs)

        example_data_path: str = kwargs["example_data_path"]
        num_examples: int = kwargs["num_examples"]

        # Set input text from data record
        input_text = self._confirm_ed_data_format(data)

        # For event detection the schema is the list of events with definitions
        # Concatenate event type definitions into a single string, with a line break
        # between each definition
        event_type_defs = self._get_event_types_defs(schema)

        # Some options for selecting similar examples may require additional keyword
        # arguments (e.g., name of sentence transformer), set these if they exist
        example_kwargs = kwargs.get("example_kwargs", {})
        # If selecting sentences based on similarity, then the target input text will
        # be required.
        example_kwargs["target_text"] = input_text
        # Get text of examples
        example_text = self._construct_examples(
            example_data_path, num_examples, **example_kwargs
        )

        # Define the JSON schema for the output
        json_schema = get_ed_json_schema(schema)

        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "These are the possible event types you can chose from:\n"
                + event_type_defs
                + "\nHere are some examples:\n"
                + example_text
                + "\nGiven this text:\n```"
                + input_text
                + "```\n"
                "Do any of the event types appear in the text? "
                "Not all event types will appear in the text, there may be some "
                "that appear multiple times and there may be no event types "
                "in the text.\n"
                "If you identify one or more event type, then what is the trigger "
                "word or words in the input text?"
                + "\nFormat the output strictly as JSON. The output is an array of "
                "JSON objects with exactly one key-value pair. The valid keys are "
                "the event types and the values are the trigger words from the input "
                "text."
            ),
            json_schema,
            self._get_ids(data),
        )


class EDPrompter4(EDPrompter3):
    """
    Class to construct prompts for event trigger detection:
    - uses SIMILAR samples in the prompt the prompt
    - similarity is determined using a sentence transformer
    - requests JSON output with a defined JSON schema
    """

    def _get_examples(
        self, example_data_path: str, num_examples: int, **kwargs: Any
    ) -> list[dict]:
        """
        Selects a subset of example records from a dataset based on the cosine
        similarity given by a sentence transformer model.

        This function loads example data from a given file, sets a random seed for
        reproducibility, and selects `num_examples` random examples without replacement.

        Args:
            example_data_path (str): Path to the file containing example records.
            num_examples (int): The number of examples to randomly select.
            **kwargs: Additional keyword arguments, must include:
                - sentence_transformer_model (str):
                  The name of the sentence transformer model to use for similarity
                  comparison.
                - target_text (str):
                  The target text that event triggers are to be extracted from

        Returns:
            list[dict]: A list of randomly selected example records.
        """
        required_keys = {"sentence_transformer_model": str, "target_text": str}

        self._check_required_keys(required_keys, kwargs)

        model_name = kwargs["sentence_transformer_model"]

        # Check that the model name is valid by trying to load it
        # TODO: Would be more efficient to load the model once and encode the examples
        # once (in the evaluator?) and then pass these to the prompt, rather than
        # loading for each prompt.
        try:
            model = SentenceTransformer(
                model_name, similarity_fn_name=SimilarityFunction.COSINE
            )
        except Exception as e:
            err_msg = f"Invalid sentence transformer model name: {model_name}"
            raise ValueError(err_msg) from e

        target_text = kwargs["target_text"]
        target_embedding = model.encode(target_text)
        example_data = load_data(example_data_path)
        example_embeddings = model.encode(example_data)

        # Compute cosine similarity between the target and all text records
        cos_scores = model.similarity(target_embedding, example_embeddings)[0]

        # Get the indices of the top 3 most similar records
        top_results = torch.topk(cos_scores, k=num_examples)

        selected_examples = []
        for _, idx in zip(top_results[0], top_results[1], strict=False):
            selected_examples.append(example_data[idx])

        return selected_examples


class EAEPrompter1(EEPrompt):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - does NOT use examples in the prompt
    - requests JSON output
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_ids(self, data: dict[str, str | int | list]) -> dict[str, str]:
        """
        Retrieves the IDs to be attached to the LLM's response to ensure
        the record can be identified.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the data record.

        Returns:
            dict[str, str]: A dictionary containing the IDs for the record.
        """
        if "event_mentions" not in data:
            err_msg = "Key 'event_mentions' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(data["event_mentions"], list):
            err_msg = (
                "Expected 'event_mentions' to be a list, but got "
                f"{type(data['event_mentions']).__name__}."
            )
            raise TypeError(err_msg)
        if "id" not in data["event_mentions"][0]:
            err_msg = "Key 'id' is missing from the event mention record."
            raise KeyError(err_msg)
        if not isinstance(data["event_mentions"][0]["id"], str):
            err_msg = (
                "Expected 'id' to be a str, but got "
                f"{type(data['event_mentions'][0]['id']).__name__}."
            )
            raise TypeError(err_msg)
        return {"evt_id": data["event_mentions"][0]["id"]}

    def _get_system_content(self) -> str:
        return (
            "You are an assistant that extracts event arguments from input text. "
            "Event arguments are the key entities, participants, and attributes "
            "associated with an event in a text. They provide context and specific "
            "details about what happened, who was involved, and where it occurred. "
            "\n"
            "---\n"
        )

    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        Args:
            data (dict[str, str | int | list]):
                A dictionary containing the data record.
            schema (dict):
                A dictionary of the possible event types and their definitions.
            **kwargs:
                Additional keyword arguments.

        Returns:
            tuple[str, str]: A tuple containing:

            - str: The constructed prompt.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """

        event_mention, tokens = self._confirm_arg_data_format(data)

        # Get trigger span for 1st event mention (should be only event mention)
        trig_start, trig_end, event_type = self._get_event_trigger(event_mention)

        # Get trigger based on tagged tokens
        input_text = self._get_tagged_text(tokens, trig_start, trig_end)

        # Get additional details for the prompt
        arg_roles = get_arg_roles(schema, event_type)
        number_of_args_str = (
            "There will be at most one argument for each role "
            "and there may be no arguments for a given role. "
        )

        return (
            (
                self._get_system_content()
                + "An event type of "
                + event_type
                + " has been detected in the input "
                "text. The trigger word is identified with a <trigger> tag.\n"
                "Input text:\n```" + input_text + "```\n"
                "What are the words in the input text that represent the following "
                "argument roles? \n"
                + "\n".join(arg_roles)
                + "\n"
                + number_of_args_str
                + "\nFormat the output strictly as JSON, ensuring each argument "
                "role has a corresponding key. "
                "If an argument is not explicitly mentioned, return `null` for "
                "that role.\n"
            ),
            "json",
            self._get_ids(data),
        )


class EAEPrompter2(EAEPrompter1):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - does NOT use examples in the prompt
    - requests JSON output and gives a specific schema
    """

    def __init__(self) -> None:
        super().__init__()

    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: Any,
    ) -> tuple[str, str | dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        Args:
            data (dict[str, str | int]):
                A dictionary containing the data record.
            schema (dict):
                A dictionary of the possible event types and their definitions.
            **kwargs:
                Additional keyword arguments

        Returns:
            tuple[str, str]: A tuple containing:

            - str: The constructed prompt.
            - str | dict: The output format, either 'json' or a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """

        event_mention, tokens = self._confirm_arg_data_format(data)

        # Get trigger span for 1st event mention (should be only event mention)
        trig_start, trig_end, event_type = self._get_event_trigger(event_mention)

        # Get trigger based on tagged tokens
        input_text = self._get_tagged_text(tokens, trig_start, trig_end)

        # Get additional details for the prompt
        arg_roles = get_arg_roles(schema, event_type)
        number_of_args_str = (
            "There will be at most one argument for each role "
            "and there may be no arguments for a given role. "
        )

        return (
            (
                self._get_system_content()
                + "An event type of "
                + event_type
                + " has been detected in the input "
                "text. The trigger word is identified with a <trigger> tag.\n"
                "Input text:\n```" + input_text + "```\n"
                "What are the words in the input text that represent the following "
                "argument roles? \n"
                + "\n".join(arg_roles)
                + "\n"
                + number_of_args_str
                + "\nFormat the output strictly as JSON, ensuring each argument "
                "role has a corresponding key. "
                "If an argument is not explicitly mentioned, return `null` for "
                "that role.\n"
            ),
            pattern_to_json_schema(event_type, arg_roles),
            self._get_ids(data),
        )
