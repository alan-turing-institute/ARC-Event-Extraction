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

        if len(arg_roles) != len(set(arg_roles)):
            # There could be more than one argument per role
            number_of_args_str = ""
            for role in set(arg_roles):
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
        tokens: list[str],
        event_mentions: list[dict],
        schema: dict,
    ) -> tuple[str, str]:
        """
        This method will construct a prompt given a list of tokens, a list of
        dictionary event mentions (assumed to be the same format as the processed
        data from TextEE) and a json schema.
        """

    @abstractmethod
    def get_prompt(
        self,
        tokens: list[str],
        event_mentions: list[dict],
        schema: dict,
        **kwargs: str | int,
    ) -> tuple[str, str]:
        """
        This method returns a prompt given a list of tokens, a list of
        dictionary event mentions (assumed to be the same format as the processed
        data from TextEE) and a json schema.
        The event mention data might not be used by all classes that inherit (e.g., if
        only doing event detection)
        The kwargs may contain other required data, such as a file path for example
        dataset
        """
