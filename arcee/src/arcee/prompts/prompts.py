from __future__ import annotations

from abc import ABC, abstractmethod


class EAEPrompt(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_system_content(self) -> str:
        pass

    @abstractmethod
    def get_prompt(
        self,
        tokens: list[str],
        event_mention: dict,
        pattern: dict,
    ) -> tuple[str, str]:
        """
        to do
        """


class EAEPromptNoExample_01(EAEPrompt):
    """
    To do: Event argument extraction for a SINGLE event trigger

    """

    def __init__(self) -> None:
        super().__init__()

    def get_system_content(self) -> str:
        return (
            "You are an assistant that helps extract the list of arguments for an "
            "event from input text."
        )

    def get_prompt(
        self,
        tokens: list[str],
        event_mentions: dict,
        pattern: dict,
    ) -> tuple[str, str]:
        """
        To do
        Returns both the prompt and the expected format of the response, e.g. empty
        string (for no constraint) and 'json' for JSON format
        """
        event_mention = event_mentions[0]
        trig_start = event_mention["trigger"]["start"]
        trig_end = event_mention["trigger"]["end"]  # End index is exclusive
        tokens[trig_start] = "<trigger>" + tokens[trig_start]
        tokens[trig_end - 1] = tokens[trig_end - 1] + "</trigger>"

        input_text = " ".join(tokens)

        event_type = event_mention["event_type"]

        arg_roles = pattern[event_type]["roles"]

        prompt = (
            "An event type of " + event_type + " has been detected in the input text. "
            "The trigger word is identified with a <trigger> tag.\n "
            "Input text:\n```" + input_text + "```\n "
            "What are the words in the input text that represent the argument role "
            "```" + arg_roles[0] + "```?"
        )

        return prompt, ""
