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
        examples: list[dict],
    ) -> tuple[str, str]:
        """
        This method creates a prompt given input text, event mention data, a
        schema and a list of examples.
        The event mention data might not be used by all classes that inherit (e.g., if
        only doing event detection)
        The examples may not be used by all classes
        """


class EAEPromptNoExample_01(EEPrompt):
    """
    Event argument extraction:
    - only takes a SINGLE event trigger
    - only predicts ONE arguement role at a time
    - does NOT use examples in the prompt
    - returns natural language prompt, rather than structured format

    This class might only be suitable for exploratory work as the natural language
    will need to be parsed by a human.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_prompt(
        self,
        tokens: list[str],
        event_mentions: list,
        schema: dict,
        examples: list[dict],
    ) -> tuple[str, str]:
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
        examples: list[dict]
            A list of example records in dictionary form, expected to be an empty string
            for this class but could be used by other classes to sample records to
            include in the prompt

        Returns
        -------
        tuple[str, str]:
            The first string is the prompt and the second is the response format,
            e.g., 'json'
        """
        if len(examples) > 0:
            warning_msg = (
                "Example records have been provided to a prompt that does "
                "not use them."
            )
            raise Warning(warning_msg)
        event_mention = event_mentions[0]
        trig_start = event_mention["trigger"]["start"]
        trig_end = event_mention["trigger"]["end"]  # End index is exclusive
        tokens[trig_start] = "<trigger>" + tokens[trig_start]
        tokens[trig_end - 1] = tokens[trig_end - 1] + "</trigger>"

        input_text = " ".join(tokens)

        event_type = event_mention["event_type"]

        arg_roles = schema[event_type]["roles"]

        prompt = (
            "An event type of " + event_type + " has been detected in the input text. "
            "The trigger word is identified with a <trigger> tag.\n "
            "Input text:\n```" + input_text + "```\n "
            "What are the words in the input text that represent the argument role "
            "```" + arg_roles[0] + "```?"
        )

        # Want this to return natural language so specify response format as an empty
        # string (will work for Ollama - might need to update when using HF?)
        return prompt, ""
