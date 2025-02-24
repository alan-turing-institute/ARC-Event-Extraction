from __future__ import annotations

from abc import ABC, abstractmethod


class EEPrompt(ABC):
    """
    Abstract class for prompt construction. This will define common functionality
    that child prompt classes will inherit.
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_system_content(self) -> str:
        """
        This method will return the system content message to be used
        This could be overwritten by a child class.
        """
        return (
            "You are an assistant that helps conducts event extraction "
            "from input text.\n"
        )

    def _confirm_data_format(
        self, data: dict[str, str | int | list]
    ) -> tuple[dict, list]:
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
        if "tokens" not in data:
            err_msg = "Key 'tokens' is missing from the data."
            raise KeyError(err_msg)
        if not isinstance(data["tokens"], list):
            err_msg = (
                "Expected 'tokens' to be a string, but got "
                f"{type(data['tokens']).__name__}."
            )
            raise TypeError(err_msg)

        return data["event_mentions"][0], data["tokens"]

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
        Inserts trigger tags into a list of tokens given the start and end
        indices, and returns the modified text.

        Parameters
        ----------
        tokens : list of str
            A list of tokens representing a text sequence.
        trig_start : int
            The starting index of the trigger span.
        trig_end : int
            The exclusive ending index of the trigger span.

        Returns
        -------
        str
            The modified text with `<trigger>` and `</trigger>` tags
            surrounding the trigger span.

        Raises
        ------
        ValueError
            If trig_start is not less than trig_end or if trig_end exceeds the list
            bounds.
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

    def _get_arg_schema_description(self, arg_roles: list[str]) -> str:
        """
        Generates a textual description of the argument schema based on role
        occurrences. This can be used to add to a prompt to describe a schema
        if the model isn't able to take the JSON schema as an output format.

        Parameters
        ----------
        arg_roles : list[str]
            A list of argument roles, where each role may appear multiple times.

        Returns
        -------
        str
            A description of the argument schema, indicating the possible number of
            arguments for each role.
        """
        number_of_args_str = (
            "There will be at most one argument for each role "
            "and there may be no arguments for a given role. "
        )
        unique_arg_roles = set(arg_roles)
        if len(arg_roles) != len(unique_arg_roles):
            # There could be more than one argument per role
            number_of_args_str = ""
            for role in unique_arg_roles:
                number_of_args_str += (
                    "There may be between zero and "
                    + str(arg_roles.count(role))
                    + " arguments "
                    "for role " + role + ". "
                )

        return number_of_args_str

    @abstractmethod
    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
    ) -> tuple[str, str]:
        """
        Abstract method for constructing the prompt
        """

    def get_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
        **kwargs: str | int,
    ) -> tuple[str, str]:
        """
        Method to return a prompt, this may get overwritten by child classes

        Parameters
        ----------
        tokens: list[str]
            The input text in token form
        schema: dict[str, str]
            A schema which is a dictionary. For event type detection, this
            dictionary will contain the event types and their definitions.
            For argument extraction, it will contain the event types and
            the possible arguemnt roles for each event type.

        Returns
        -------
        str, str
            Returns the prompt and the output format
        """

        prompt, output_format = self._construct_prompt(data, schema)

        return prompt, output_format


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
        """
        return (
            "You are an assistant that extracts event triggers and their associated "
            "event types from input text. Many triggers are verbs and represent a "
            "change of state or action in the text."
            "\n"
            "---\n"
        )

    def _construct_prompt(
        self,
        data: dict[str, str | int | list],
        schema: dict,
    ) -> tuple[str, str]:
        """
        Construct a prompt for event trigger detection.

        Parameters
        ----------
        data : dict[str, str | int]
            A dictionary containing the data record.
        schema: dict
            A dictionary of the possible event types and their definitions.

        Returns
        -------
        tuple[str, str]
            A tuple containing the prompt and the output format.
        """
        # Set input text from data record
        input_text = str(data["text"])  # type: str

        # For event detection the schema is the list of events with definitions
        # Concatenate event type definitions into a single string, with a line break
        # between each definition
        event_type_defs = "\n".join([key + ": " + schema[key] for key in schema])
        # Return the constructed prompt
        return (
            (
                self._get_system_content()
                + "These are the possible event types you can chose from:\n"
                + event_type_defs
                + "\nGiven this text:\n```"
                + input_text
                + "```\n"
                "Do any of the event types appear in the text?\n"
                "If you identify one or more event type, then what is the trigger "
                "word or words in the input text?"
                + "\nFormat the output strictly as JSON, ensuring the detected "
                "event types are the keys and the trigger words are the values. "
                "The values must be a string type and not an array."
            ),
            "json",
        )


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

    def _get_arg_roles(self, schema: dict, event_type: str) -> list[str]:
        """
        Retrieves the list of argument roles associated with a given event type.

        Parameters
        ----------
        schema : dict
            A dictionary mapping event types to their respective schemas,
            including argument roles.
        event_type : str
            The event type whose argument roles should be retrieved.

        Returns
        -------
        list of str
            A list of argument roles associated with the specified event type.

        Raises
        ------
        KeyError
            If the given event_type is not found in the schema.
        """
        if event_type not in schema:
            err_msg = f"Event type '{event_type}' not found in schema."
            raise KeyError(err_msg)

        return schema[event_type]["roles"]

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
    ) -> tuple[str, str]:
        """
        Construct a prompt for event trigger detection.

        Parameters
        ----------
        data : dict[str, str | int]
            A dictionary containing the data record.
        schema: dict
            A dictionary of the possible event types and their definitions.

        Returns
        -------
        tuple[str, str]
            A tuple containing the prompt and the output format.
        """

        event_mention, tokens = self._confirm_data_format(data)

        # Get trigger span for 1st event mention (should be only event mention)
        trig_start, trig_end, event_type = self._get_event_trigger(event_mention)

        # Get trigger based on tagged tokens
        input_text = self._get_tagged_text(tokens, trig_start, trig_end)

        # Get additional details for the prompt
        arg_roles = self._get_arg_roles(schema, event_type)
        number_of_args_str = self._get_arg_schema_description(arg_roles)

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
                + "\n".join(set(arg_roles))
                + "\n"
                + number_of_args_str
                + "\nFormat the output strictly as JSON, ensuring each argument "
                "role has a corresponding key. "
                "If an argument is not explicitly mentioned, return `null` for "
                "that role.\n"
            ),
            "json",
        )
