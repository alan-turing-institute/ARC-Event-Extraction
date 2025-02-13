from __future__ import annotations

import json

import guidance as gi


def get_llm_responce(
    lmodel: gi.models.Transformers,
    prompt: list[dict],
    sentence: str,
    event_list: list[str],
) -> dict:
    """
    Sends a chat request to the specified model using the provided client and prompt,
    and returns the response as a dictionary.

    Args:
        client (Client): The client instance used to communicate with the model.
        prompt (list): This can only contain one message from the user.
        sentence (str): The sentence to be parsed.
        event_list (list): A list of possible event types.

    Returns:
        dict: The response from the model, parsed as a JSON dictionary.
    """

    assert len(prompt) == 1, "Prompt must contain only one message from the user."
    assert prompt[0]["role"] == "user", "The first message must be from the user."

    with gi.user():
        lmodel += prompt[0]["content"]
    with gi.assistant():
        lmodel += '{EventType: "' + gi.select(event_list, name="EventType") + '",'
        lmodel += 'EventTrigger: "' + gi.substring(sentence, name="EventTrigger") + '",'
        lmodel += 'Subject: "' + gi.substring(sentence, name="Subject") + '",'
        lmodel += 'Object: "' + gi.substring(sentence, name="Object") + '"}'

    return lmodel._variables


def get_unguided_llm_responce(
    lmodel: gi.models.Transformers,
    prompt: list[dict],
    _sentence: str,
    _event_list: list[str],
) -> dict:
    """
    Sends a chat request to the specified model using the provided client and prompt,
    and returns the response as a dictionary.

    Args:
        client (Client): The client instance used to communicate with the model.
        prompt (list): This can only contain one message from the user.
        sentence (str): The sentence to be parsed.
        event_list (list): A list of possible event types.

    Returns:
        dict: The response from the model, parsed as a JSON dictionary.
    """

    assert len(prompt) == 1, "Prompt must contain only one message from the user."
    assert prompt[0]["role"] == "user", "The first message must be from the user."

    with gi.user():
        lmodel += prompt[0]["content"]
    with gi.assistant():
        lmodel += "```json\n" + gi.json(name="output", max_tokens=100)

    try:
        return json.loads(lmodel["output"])
    except json.JSONDecodeError:
        return {}
