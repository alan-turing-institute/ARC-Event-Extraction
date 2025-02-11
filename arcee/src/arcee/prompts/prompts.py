from __future__ import annotations

from abc import ABC, abstractmethod


class EEPrompt(ABC):
    def __init__(self) -> None:
        super().__init__()

    def get_system_content(self) -> str:
        """
        This method will return the system content message to be used
        This could be overwritten by a child class, though that may not
        be necessary.
        """
        return (
            "You are an assistant that helps extract the list of arguments for an "
            "event from input text."
        )

    @abstractmethod
    def get_prompt(
        self,
        tokens: list[str],
        event_mentions: list,
        schema: dict,
    ) -> str:
        """
        This method creates a prompt given input text, event mention data, a
        schema and a list of examples.
        The event mention data might not be used by all classes that inherit (e.g., if
        only doing event detection)
        The examples may not be used by all classes
        """


class DummyEAEPrompter(EEPrompt):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - only predicts ONE argument role at a time
    - does NOT use examples in the prompt
    - requests JSON output
    """

    def __init__(self) -> None:
        super().__init__()

    def get_prompt(
        self,
        tokens: list[str],
        event_mentions: list,
        schema: dict,
    ) -> str:
        """
        Parameters
        ----------
        tokens: list[str]
            The input text in token form
        event_mentions: list
            From the processed data, the list of event mentions for the input text.
            Multiple events may exist in for some records, but this method creates
            a prompt only using the first of these.
        schema: dict
            A schema for the dataset (currently taken from TextEE pattern files)

        Returns
        -------
        str:
            Returns the prompt
        """
        # Get trigger span for 1st event mention
        event_mention = event_mentions[0]
        trig_start = event_mention["trigger"]["start"]
        trig_end = event_mention["trigger"]["end"]  # End index is exclusive

        # Get trigger based on tagged tokens
        tagged_tokens = tokens.copy()
        tagged_tokens[trig_start] = "<trigger>" + tagged_tokens[trig_start]
        tagged_tokens[trig_end - 1] = tagged_tokens[trig_end - 1] + "</trigger>"
        input_text = " ".join(tagged_tokens)

        # Get additional details for the prompt
        event_type = event_mention["event_type"]
        arg_roles = schema[event_type]["roles"]

        return (
            "An event type of " + event_type + " has been detected in the input text. "
            "The trigger word is identified with a <trigger> tag.\n "
            "Input text:\n```" + input_text + "```\n "
            "What are the words in the input text that represent the argument role "
            "```" + arg_roles[0] + "```? Provide your response in JSON format with the "
            "key " + arg_roles[0] + " and the value as the words in the input text "
            "that represent the argument."
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

    def get_prompt(
        self,
        tokens: list[str],
        event_mentions: list,
        schema: dict,
    ) -> str:
        """
        Parameters
        ----------
        tokens: list[str]
            The input text in token form
        event_mentions: list
            From the processed data, the list of event mentions for the input text.
            Multiple events may exist in for some records, but this method creates
            a prompt only using the first of these.
        schema: dict
            A schema for the dataset (currently taken from TextEE pattern files)

        Returns
        -------
        str:
            Returns the prompt
        """
        # Get trigger span for 1st event mention
        event_mention = event_mentions[0]
        trig_start = event_mention["trigger"]["start"]
        trig_end = event_mention["trigger"]["end"]  # End index is exclusive

        # Get trigger based on tagged tokens
        tagged_tokens = tokens.copy()
        tagged_tokens[trig_start] = "<trigger>" + tagged_tokens[trig_start]
        tagged_tokens[trig_end - 1] = tagged_tokens[trig_end - 1] + "</trigger>"
        input_text = " ".join(tagged_tokens)

        # Get additional details for the prompt
        event_type = event_mention["event_type"]
        arg_roles = schema[event_type]["roles"]

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
                    + "arguments "
                    "for role " + role + ". "
                )

        return (
            "An event type of " + event_type + " has been detected in the input text. "
            "The trigger word is identified with a <trigger> tag.\n "
            "Input text:\n```" + input_text + "```\n "
            "What are the words in the input text that represent the following "
            "argument roles? "
            + ",\n".join(set(arg_roles))
            + "? "
            + number_of_args_str
            + "Provide your response in JSON format with one key for each"
            "argument role and the value as the words in the input text "
            "that represent the argument."
        )
