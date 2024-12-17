from __future__ import annotations

__all__ = ["etype_prompt", "etrig_prompt", "eall_prompt"]

default_event_types = (
    "['Clean', 'Cook', 'Travel', 'Exercise', 'Meet', 'Injure', 'Purchase', "
    "'Transport', 'Hire', 'Build']"
)


def etype_prompt(
    sentence: str,
    examples: list[dict] | None = None,
    event_types: str = default_event_types,
) -> list:
    """
    Generates a prompt for event type extraction from a given sentence.
    This prompt is only designed to extract the EventType from a sentence.

    Args:
        sentence (str): The sentence from which to extract the event type.
        examples (list[dict] | None, optional): A list of example dictionaries, where
            each dictionary contains a 'Sentence' and an 'EventType'. Defaults to None.
        event_types (str, optional): A string representation of the list of possible
            event types. Defaults to default_event_types.
    Returns:
        list: A list containing the prompt to pass to Ollama.
    """
    if examples is not None:
        example_strings = [
            f"Example: {ex['Sentence']} is {{'EventType': '{ex['EventType']}'}}"
            for ex in examples
        ]
        examples_strings = "\n".join(example_strings)
        include_examples = f"Here are a few examples:\n{examples_strings}\n"
    else:
        include_examples = ""

    return [
        {
            "role": "user",
            "content": (
                "I am going to give you a sentence and I want you to tell me what type "
                "of event it is describing. There are ten possible types of events "
                f"you can choose from. "
                f"{include_examples}"
                "Please give your answer in a json with the key 'EventType' matching "
                f"one of the following options: {event_types}. \n"
                f"The sentence is: {sentence}"
            ),
        },
    ]


def etrig_prompt(
    sentence: str,
    examples: list[dict] | None = None,
    event_types: str = default_event_types,
) -> list:
    """
    Generates a prompt for event type extraction from a given sentence. This prompt
    is only designed to extract the EventType and the EventTrigger from a sentence.

    Args:
        sentence (str): The sentence from which to extract the event type.
        examples (list[dict] | None, optional): A list of example dictionaries, where
            each dictionary contains a 'Sentence' and an 'EventType'. Defaults to None.
        event_types (str, optional): A string representation of the list of possible
            event types. Defaults to default_event_types.
    Returns:
        list: A list containing the prompt to pass to Ollama.
    """
    if examples is not None:
        example_strings = [
            (
                f"Example: {ex['Sentence']} is "
                f"{{'EventType': '{ex['EventType']}',"
                f" 'EventTrigger': '{ex['EventTrigger']}}}"
            )
            for ex in examples
        ]
        examples_strings = "\n".join(example_strings)
        include_examples = f"Here are a few examples:\n{examples_strings}\n"
    else:
        include_examples = ""

    return [
        {
            "role": "user",
            "content": (
                "I am going to give you a sentence and I want you to tell me what type "
                "of event it is describing and what is the indicating verb. There are "
                "ten possible types of events you can choose from. "
                f"{include_examples}"
                "Please give your answer in a json with the key 'EventType' matching "
                f"one of the following options: {event_types}. and the 'EventTrigger' "
                "matching the relevant verb in the sentence.\n"
                f"The sentence is: {sentence}"
            ),
        },
    ]


def eall_prompt(
    sentence: str,
    examples: list[dict] | None = None,
    event_types: str = default_event_types,
) -> list:
    """
    Generates a prompt for event type extraction from a given sentence.
    This prompt is designed to extract the EventType, EventTrigger, Subject, and Object
    from a sentence.

    Args:
        sentence (str): The sentence from which to extract the event type.
        examples (list[dict] | None, optional): A list of example dictionaries, where
            each dictionary contains a 'Sentence' and an 'EventType'. Defaults to None.
        event_types (str, optional): A string representation of the list of possible
            event types. Defaults to default_event_types.
    Returns:
        list: A list containing the prompt to pass to Ollama.
    """
    if examples is not None:
        example_strings = [
            (
                f"Example: {ex['Sentence']} is "
                f"{{'EventType': '{ex['EventType']}', "
                f"'EventTrigger': '{ex['EventTrigger']}, "
                f"'Subject': '{ex['Subject']}', "
                f"'Object': '{ex['Object']}'}}"
            )
            for ex in examples
        ]
        examples_strings = "\n".join(example_strings)
        include_examples = f"Here are a few examples:\n{examples_strings}\n"
    else:
        include_examples = ""

    return [
        {
            "role": "user",
            "content": (
                "I am going to give you a sentence and I want you to tell me what type "
                "of event it is describing and what is the indicating verb, object, "
                "and subject of the sentence. There are ten possible types of events "
                "you can choose from. "
                f"{include_examples}"
                "Please give your answer in a json with the key 'EventType' matching "
                f"one of the following options: {event_types}. The 'EventTrigger' "
                "matching the relevant verb in the sentence. The 'Subject' matching "
                "the actor in the sentence, and 'Object' matching the target object.\n"
                f"The sentence is: {sentence}"
            ),
        },
    ]
