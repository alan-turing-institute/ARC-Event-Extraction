from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

from arcee.dataset.load import load_data
from arcee.dataset.schemas import (
    data_to_argument_schema,
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

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        """
        Initialises the EEPrompt class and sets the attributes that may be required
        when constructing a prompt.

        Args:
            schema: dict
                The event schema to be used for the data
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self._schema = schema
        # The `input_data` and `input_text` attributes will be set each time the
        # `get_prompt` method is called
        self._input_data: dict[str, str | int | list] = {}
        self._input_text: str = ""
        # This will be set to 'json' by default, but can be overridden by child classes
        # that define a specific JSON schema
        self._json_schema: dict = {}
        # This will be set to False by default, and can be set to True if the class
        # is generating prompts for event argument extraction
        self._eae: bool = False

    @abstractmethod
    def _get_system_content(self) -> str:
        """
        Returns the system content message to be used.

        Returns:
            str: The system content message.
        """

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

    def _check_required_keys(
        self, required_keys: dict, kwargs: dict
    ) -> tuple[Any, ...]:
        """
        Validates that all required keys are present in the kwargs dictionary and have
        the correct types.

        Args:
            required_keys (dict):
                A dictionary mapping required keyword argument names to their expected
                types.
            kwargs (dict):
                A dictionary containing keyword arguments to be validated.

        Returns:
            tuple[Any, ...]: The values of the required keys in the order they
            appear in `required_keys`.

        Raises:
            TypeError: If a required key is missing from `kwargs`.
            TypeError: If a required key has a value of an incorrect type.
        """
        values = []
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
            values.append(kwargs[key])
        return tuple(values)

    @abstractmethod
    def _get_ids(self) -> dict[str, str]:
        """
        Retrieves the IDs for the current 'input_data' record to be attached to the
        LLM's response for record identification.

        Returns:
            dict[str, str]: A dictionary containing the IDs for the record.
        """

    @abstractmethod
    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Abstract method for constructing the prompt.

        Returns:
            tuple[str, dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - dict: The output format as a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response that
                    identify the record.
        """

    def get_prompt(
        self, data: dict[str, str | int | list]
    ) -> tuple[str, dict, dict, str]:
        """
        Returns a prompt for a given data record.

        Args:
            data (dict[str, str | int | list]):
                The input data record.

        Returns:
            tuple[str, str | dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - dict: The output format as a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
            that identify the record.
            - str: The input text

        """
        # Set the `input_data` and `input_text` attributes, now that we have a data
        # record to construct a prompt for
        self._input_data = data
        self._input_text = self._confirm_ed_data_format(data)
        # Call the `_construct_prompt` method to generate the prompt
        prompt, output_format, ids = self._construct_prompt()
        return prompt, output_format, ids, self._input_text


class EDPrompt(EEPrompt, ABC):
    """
    Abstract class for event detection prompt construction. This will define common
    functionality that child prompt classes for event detection will inherit.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        # For event detection the schema is the list of events with definitions
        # Concatenate event type definitions into a single string, with a line break
        # between each definition
        self._event_type_defs = self._get_event_types_defs()
        # Define the JSON schema for the output
        self._json_schema = get_ed_json_schema(self._schema)

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

    def _get_event_types_defs(self) -> str:
        """
        Retrieves event types and their definitions from the schema and returns them as
        a string.

        Returns:
            str: A string containing the event type definitions.
        """

        return "\n".join([key + self._schema[key] for key in self._schema])

    def _get_ids(self) -> dict[str, str]:
        """
        Retrieves the window id for the current `input_data` attributed to be attached
        to the LLM's response for record identification.

        Returns:
            dict[str, str]: A dictionary containing the window ID (`wnd_id`).

        Raises:
            KeyError: If the `wnd_id` key is missing from the data.
            TypeError: If the value of `wnd_id` is not a string.
        """
        if "wnd_id" not in self._input_data:
            err_msg = "Key 'wnd_id' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(self._input_data["wnd_id"], str):
            err_msg = (
                "Expected 'wnd_id' to be a str, but got "
                f"{type(self._input_data['wnd_id']).__name__}."
            )
            raise TypeError(err_msg)
        return {"wnd_id": self._input_data["wnd_id"]}

    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        Returns:
            tuple[str, dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - dict: The output format as a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """
        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "These are the possible event types you can chose from:\n"
                + self._event_type_defs
                + "\nGiven this input text:\n```"
                + self._input_text
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
            self._json_schema,
            self._get_ids(),
        )


class EDPromptWithExampleSchema(EDPrompt, ABC):
    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        # To be correct JSON schema, the keys & values have to be in double quotes
        self._example_schema = (
            '[{"EventTypeA": "trigger1"}, {"EventTypeB": "trigger words2"}, '
            '{"EventTypeA": "trigger words3"}]'
        )

    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        Returns:
            tuple[str, dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - dict: The output format as a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """
        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "You will be give some text and have to determine if any event types "
                "appear in the text. "
                "Not all event types will appear, there may be some "
                "that appear multiple times and there may be no event types "
                "in the text.\n"
                "If you identify one or more event type, then what is the trigger "
                "word or words in the input text?"
                + "\nFormat the output strictly as JSON. The output is an array of "
                "JSON objects with exactly one key-value pair. The valid keys are "
                "the event types and the values are the trigger words from the input "
                "text. An example of the schema is:\n"
                + self._example_schema
                + "\n\n"
                + "These are the possible event types you can chose from:\n"
                + self._event_type_defs
                + "\nNow process this input text:\n```"
                + self._input_text
                + "```\n"
                + "Response (JSON): "
            ),
            self._json_schema,
            self._get_ids(),
        )


class EAEPrompt(EEPrompt, ABC):
    """
    Abstract class for event argument extraction prompt construction. This will define
    common functionality that child prompt classes will inherit.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        self._eae = True
        # Some attributes specific to argument extraction
        self._tagged_text: str = ""
        self._arg_roles: list[str] = []
        self._event_type: str = ""

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

    def _get_event_trigger(
        self, event_mention: dict[str, dict[str, int] | str]
    ) -> tuple[int, int, str]:
        """
        Extracts the trigger information from an event mention.
        The event mention is expected to be in the same format as the processed
        data from TextEE.

        Args:
            event_mention : dict[str, int | str]
                A dictionary containing event mention data with a "trigger" and
                "event_type" keys.

        Returns:
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

    def _get_ids(self) -> dict[str, str]:
        """
        Retrieves the event id for the current `input_data` record to be attached to the
        LLM's response to ensure the record can be identified.

        Returns:
            dict[str, str]: A dictionary containing the IDs for the record.
        """
        if "event_mentions" not in self._input_data:
            err_msg = "Key 'event_mentions' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(self._input_data["event_mentions"], list):
            err_msg = (
                "Expected 'event_mentions' to be a list, but got "
                f"{type(self._input_data['event_mentions']).__name__}."
            )
            raise TypeError(err_msg)
        if "id" not in self._input_data["event_mentions"][0]:
            err_msg = "Key 'id' is missing from the event mention record."
            raise KeyError(err_msg)
        if not isinstance(self._input_data["event_mentions"][0]["id"], str):
            err_msg = (
                "Expected 'id' to be a str, but got "
                f"{type(self._input_data['event_mentions'][0]['id']).__name__}."
            )
            raise TypeError(err_msg)
        return {"evt_id": self._input_data["event_mentions"][0]["id"]}

    def _get_system_content(self) -> str:
        """
        Returns the system content message to be used.

        Returns:
            str: The system content message
        """
        return (
            "You are an assistant that extracts event arguments from input text. "
            "Event arguments are the key entities, participants, and attributes "
            "associated with an event in a text. They provide context and specific "
            "details about what happened, who was involved, and where it occurred. "
            "\n"
            "---\n"
        )

    def _set_vars_for_prompt(self) -> None:
        """
        Sets the variables required for constructing the prompt using the attributes
        of the class instance.
        """
        event_mention, tokens = self._confirm_arg_data_format(self._input_data)

        # Get trigger span for 1st event mention (should be only event mention)
        trig_start, trig_end, self._event_type = self._get_event_trigger(event_mention)

        # Get trigger based on tagged tokens
        self._tagged_text = self._get_tagged_text(tokens, trig_start, trig_end)

        # Get additional details for the prompt
        self._arg_roles = get_arg_roles(self._schema, self._event_type)

        # Set for each input data record as it will depend on the event type
        self._json_schema = pattern_to_json_schema(self._event_type, self._arg_roles)

    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Constructs a prompt for event argument extraction.

        Returns:
            tuple[str, dict, dict]: A tuple containing:

            - str: The constructed prompt.
            - dict: The output format as a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """
        self._set_vars_for_prompt()

        return (
            (
                self._get_system_content()
                + "An event type of "
                + self._event_type
                + " has been detected in the input "
                "text. The trigger word is identified with a <trigger> tag.\n"
                "Input text:\n```" + self._tagged_text + "```\n"
                "What are the words in the input text that represent the following "
                "argument roles? \n"
                + "\n".join(self._arg_roles)
                + "\n"
                + "There will be at most one argument for each role "
                "and there may be no arguments for a given role. "
                + "\nFormat the output strictly as JSON, ensuring each argument "
                "role has a corresponding key. "
                "If an argument is not explicitly mentioned, return `null` for "
                "that role.\n"
            ),
            self._json_schema,
            self._get_ids(),
        )


class EAEPromptWithExampleSchema(EAEPrompt, ABC):
    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        # To be correct JSON schema, the keys & values have to be in double quotes
        self._example_schema = (
            '{"role1": "arg1", "role2": null, "role3": null, "role4": "arg4"}'
        )

    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Constructs a prompt for event argument extraction using an example of the
        required schema.

        Returns:
            tuple[str, dict, dict]: A tuple containing:

            - str: The prompt to be used in the task.
            - dict: The output format as a JSON schema.
            - dict: A dictionary of IDs to be attached to the LLM's response
              that identify the record.
        """
        self._set_vars_for_prompt()

        return (
            (
                self._get_system_content()
                + "An event type of "
                + self._event_type
                + " has been detected in the input "
                "text. The trigger word is identified with a <trigger> tag.\n"
                "Input text:\n```" + self._tagged_text + "```\n"
                "What are the words in the input text that represent the following "
                "argument roles? \n"
                + "\n".join(self._arg_roles)
                + "\n"
                + "There will be at most one argument for each role "
                "and there may be no arguments for a given role. "
                + "\nFormat the output strictly as JSON, ensuring each argument "
                "role has a corresponding key. "
                "If an argument is not explicitly mentioned, return `null` for "
                "that role. An example of the required schema is:\n"
                + self._example_schema
                + "\n"
            ),
            self._json_schema,
            self._get_ids(),
        )


class EEPromptWithExamples(EEPrompt, ABC):
    """
    Abstract class for creating a prompt with examples in it. This class will define
    the common functionality that child classes will inherit, which could be classes
    for both event trigger detection and event argument extraction.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)

        # Define the attributes that may be used when constructing a prompt
        self._example_embeddings: list = []
        self._rng: np.random.Generator | None = None
        self._model: SentenceTransformer | None = None
        self._random_selection: bool = False
        self._same_event_type: bool = False
        self._examples_for_prompt: str = ""
        self._event_type = ""

        # Check required keys exist and the values have expected types.
        required_keys = {"example_data_path": str, "num_examples": int}
        example_args: tuple[str, int] = self._check_required_keys(required_keys, kwargs)
        example_data_path, self._num_examples = example_args

        # Load data for examples, if doing EAE will load one record per event type,
        # otherwise will load all records
        self._example_data = load_data(example_data_path, record_per_event=self._eae)

    def _get_examples(self) -> list[dict]:
        """
        Method for retrieving examples to be used in the prompt

        Returns:
            list[dict]: A list of examples to be used in the prompt

        Raises:
            ValueError: If the random number generator needed and is None
            ValueError: If the sentence transformer model is needed and is None
        """
        example_data = self._example_data
        example_embeddings = self._example_embeddings

        if self._same_event_type:
            example_data, example_embeddings = self._filter_examples_by_event_type()

        if self._random_selection:
            # Check that self._rng is set to be a random number generator
            if self._rng is None:
                err_msg = "Random number generator not set."
                raise ValueError(err_msg)
            # Return 'num_examples' random examples from the data
            return list(
                self._rng.choice(example_data, self._num_examples, replace=False)
            )
        # Selecting based on model similarity
        if self._model is None:
            err_msg = "Sentence transformer model not set."
            raise ValueError(err_msg)

        # Embed the input text
        target_embedding = self._model.encode(self._input_text)

        # Compute cosine similarity between the target and all text records
        cos_scores = self._model.similarity(target_embedding, example_embeddings)[0]

        # Get the indices of the top `num_examples` most similar records
        top_results = torch.topk(cos_scores, k=self._num_examples)

        selected_examples = []
        for _, idx in zip(top_results[0], top_results[1], strict=False):
            selected_examples.append(example_data[idx])

        return selected_examples

    def _filter_examples_by_event_type(self) -> tuple[list, list]:
        """
        A method that filters the examples by the event type of the input data record.

        Returns:
            tuple[list, list]:
                A tuple containing:
                - list: The filtered example data
                - list: The filtered example embeddings (which could be empty if the
                    random selection is being used)
        """
        filtered_indices = [
            i
            for i, record in enumerate(self._example_data)
            if record["event_mentions"][0]["event_type"] == self._event_type
        ]

        if len(filtered_indices) < self._num_examples:
            return (self._example_data, self._example_embeddings)

        filtered_example_data = [self._example_data[i] for i in filtered_indices]
        filtered_example_embeddings = []
        if not self._random_selection:
            # Checking by similarity, so the embeddings will exist for the example data
            filtered_example_embeddings = [
                self._example_embeddings[i] for i in filtered_indices
            ]
        return (filtered_example_data, filtered_example_embeddings)


class EEPromptRandomExamples(EEPromptWithExamples, ABC):
    """
    Abstract class that defines the functionality for selecting random examples to
    use in the prompt. This class could be inherited by classes for both event trigger
    detection and event argument extraction.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        # Ensure random selection will occur:
        self._random_selection = True
        # Set the key required for random selection
        required_keys = {"seed": int}
        prompt_args: tuple[int] = self._check_required_keys(required_keys, kwargs)
        seed = prompt_args[0]
        # Initialise the random number generator
        self._rng = np.random.default_rng(seed)


class EEPromptSimilarExamples(EEPromptWithExamples, ABC):
    """
    Abstract class that defines the functionality for selected examples based on
    similarity to the input text. This class could be inherited by classes for both
    event trigger detection and event argument extraction.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        # Selecting based on similarity, so set random selection to False
        self._random_selection = False
        # The name of the sentence transformer model is required for this class
        required_keys = {"sentence_transformer_model": str}
        prompt_args: tuple[str] = self._check_required_keys(required_keys, kwargs)
        model_name = prompt_args[0]

        # Load the model
        self._model = self._load_model(model_name)

        # Embed the example text for use later
        example_text = [record["text"] for record in self._example_data]
        self._example_embeddings = self._model.encode(example_text)

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """
        Check that the model name is valid by trying to load it and return the mdoel

        Args:
            model_name (str):
                Name of sentence transformer model

        Returns:
            SentenceTransformer:
                The loaded model

        Raises:
            ValueError:
                If the name of the model is invalid.
        """
        try:
            return SentenceTransformer(
                model_name, similarity_fn_name=SimilarityFunction.COSINE
            )
        except Exception as e:
            err_msg = f"Invalid sentence transformer model name: {model_name}"
            raise ValueError(err_msg) from e


class EDPromptWithExamples(EEPromptWithExamples, EDPrompt, ABC):
    """
    An abstract class for event trigger detection with examples. This class will define
    the common functionality that child classes will inherit.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        # Never filter by same event type for event trigger detection
        self._same_event_type = False
        # Always going to use the JSON dictionary schema for the output
        self._json_schema = get_ed_json_schema(self._schema)

    def _construct_examples(self, example_records: list[dict]) -> str:
        """
        Takes a list of example records and returns them in a string to be used in
        the prompt.

        Args:
            example_records (list[dict]):
                A list of the example records, each record is a dictionary

        Returns:
            str:
                Text that can be used in the prompt.
        """
        expected_output = data_to_ed_schema(example_records)

        output = []
        for i, record in enumerate(example_records):
            example_input_text = self._confirm_ed_data_format(record)

            prompt_text = (
                "### Example " + str(i + 1) + "\n"
                "Input text: \n```" + example_input_text + "```\n"
                "Response (JSON): \n"
                + json.dumps(expected_output[i]["output"], indent=4, sort_keys=True)
                + "\n"
            )
            output.append(prompt_text)

        return "\n".join(output)

    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Constructs a prompt for event trigger detection.

        This function generates a structured prompt for event detection by
        combining input text, event type definitions, and example texts. It
        also defines the expected output format using a JSON schema.

        Returns:
            tuple[str, dict, dict]: A tuple containing:
                - str: The constructed prompt with event type definitions,
                  examples, and input text.
                - dict: The JSON schema defining the expected output format.
                - dict: A dictionary containing extracted IDs from the input data.
        """
        # Get example records
        example_records = self._get_examples()
        # Turn the selected examples into the text to use for the prompt.
        self._examples_for_prompt = self._construct_examples(example_records)
        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "You will be give some text and have to determine if any event types "
                "appear in the text. "
                "Not all event types will appear, there may be some "
                "that appear multiple times and there may be no event types "
                "in the text.\n"
                "If you identify one or more event type, then what is the trigger "
                "word or words in the input text?"
                + "\nFormat the output strictly as JSON. The output is an array of "
                "JSON objects with exactly one key-value pair. The valid keys are "
                "the event types and the values are the trigger words from the input "
                "text.\n\n"
                + "These are the possible event types you can chose from:\n"
                + self._event_type_defs
                + "\nHere are some examples:\n"
                + self._examples_for_prompt
                + "\nNow process this input text:\n```"
                + self._input_text
                + "```\n"
                + "Response (JSON): "
            ),
            self._json_schema,
            self._get_ids(),
        )


class EAEPromptWithExamples(EEPromptWithExamples, EAEPrompt, ABC):
    """
    Abstract class for event argument extraction with examples. This class will define
    the common functionality that child classes will inherit.
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)

    def _construct_examples(self, example_records: list[dict]) -> str:
        """
        Takes a list of example records and returns them in a string to be used in
        the prompt.

        Args:
            example_records (list[dict]):
                A list of the example records, each record is a dictionary

        Returns:
            str:
                Text that can be used in the prompt.
        """
        expected_output = data_to_argument_schema(example_records, schema=self._schema)

        output = []
        for i, record in enumerate(example_records):
            event_mention, tokens = self._confirm_arg_data_format(record)

            # Get trigger span for 1st event mention (should be only event mention)
            trig_start, trig_end, event_type = self._get_event_trigger(event_mention)

            # Get trigger based on tagged tokens
            tagged_text = self._get_tagged_text(tokens, trig_start, trig_end)

            # Get additional details for the prompt
            arg_roles = get_arg_roles(self._schema, event_type)

            prompt_text = (
                "### Example " + str(i + 1) + "\n"
                "Event type: " + event_type + "\n"
                "Argument roles: " + ", ".join(set(arg_roles)) + "\n"
                "Input text: \n```" + tagged_text + "```\n"
                "Response (JSON): \n"
                + json.dumps(expected_output[i]["output"], indent=4, sort_keys=True)
                + "\n"
            )
            output.append(prompt_text)

        return "\n".join(output)

    def _construct_prompt(self) -> tuple[str, dict, dict]:
        """
        Constructs a prompt for event argument extraction.

        This function generates a structured prompt for event argument extraction by
        combining input text, event type definitions, and example texts. It
        also defines the expected output format using a JSON schema.

        Returns:
            tuple[str, dict, dict]: A tuple containing:
                - str: The constructed prompt with event type definitions,
                  examples, and input text.
                - dict: The JSON schema defining the expected output format.
                - dict: A dictionary containing extracted IDs from the input data.
        """
        self._set_vars_for_prompt()

        # Set the variables to use to select random examples
        example_records = self._get_examples()
        # Turn the selected examples into the text to use for the prompt.
        self._examples_for_prompt = self._construct_examples(example_records)

        return (
            (
                self._get_system_content()
                + "\n## Examples\n"
                + self._examples_for_prompt
                + "\n---\n\n## Now process the following input:\n"
                "Event type: " + self._event_type + "\n"
                "Argument roles: " + ", ".join(set(self._arg_roles)) + "\n"
                "Input text: ```" + self._tagged_text + "```\n"
                "Response (JSON):"
            ),
            self._json_schema,
            self._get_ids(),
        )


class EDPrompter1(EDPrompt):
    """
    Class to construct prompts for event trigger detection:
    - does NOT use any examples in the prompt
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)


class EDPrompter2(EDPromptWithExampleSchema):
    """
    Class to construct prompts for event trigger detection:
    - does NOT use any examples in the prompt
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)


class EDPrompter3(EEPromptRandomExamples, EDPromptWithExamples):
    """
    Class to construct prompts for event trigger detection:
    - uses RANDOM samples in the prompt the prompt, a different selection of examples
        are selected for each record
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)


class EDPrompter4(EEPromptSimilarExamples, EDPromptWithExamples):
    """
    Class to construct prompts for event trigger detection:
    - uses SIMILAR samples in the prompt the prompt
    - similarity is determined using a sentence transformer
    """

    def __init__(self, schema: dict, **kwargs: Any):
        super().__init__(schema, **kwargs)


class EAEPrompter1(EAEPrompt):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - does NOT use examples in the prompt
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema)


class EAEPrompter2(EAEPrompt):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - does NOT use examples in the prompt
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)


class EAEPrompter3(EAEPromptWithExamples, EEPromptRandomExamples):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - Uses RANDOM examples in the prompt
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        self._same_event_type = False


class EAEPrompter4(EAEPromptWithExamples, EEPromptRandomExamples):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - Uses RANDOM examples in the prompt, but with SAME EVENT TYPE
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        self._same_event_type = True


class EAEPrompter5(EAEPromptWithExamples, EEPromptSimilarExamples):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - Uses SIMILAR examples in the prompt (ANY EVENT TYPE)
    """

    def __init__(self, schema: dict, **kwargs: Any):
        super().__init__(schema, **kwargs)
        self._same_event_type = False


class EAEPrompter6(EAEPromptWithExamples, EEPromptSimilarExamples):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - predicts multiple argument roles
    - Uses SIMILAR examples in the prompt, but with SAME EVENT TYPE
    """

    def __init__(self, schema: dict, **kwargs: Any) -> None:
        super().__init__(schema, **kwargs)
        self._same_event_type = True
