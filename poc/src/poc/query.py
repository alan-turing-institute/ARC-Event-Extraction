from __future__ import annotations

import json

from ollama import Client


def get_ollama_responce(client: Client, model_choice: str, prompt: list) -> dict:
    """
    Sends a chat request to the specified model using the provided client and prompt,
    and returns the response as a dictionary.

    Args:
        client (Client): The client instance used to communicate with the model.
        model_choice (str): The identifier for the model to be used.
        prompt (list): A list of messages to be sent as the prompt to the model.

    Returns:
        dict: The response from the model, parsed as a JSON dictionary.
    """
    responce = client.chat(
        model_choice,
        messages=prompt,
        options={"Temperature": 0.5},  # This option can pass many settings
        format="json",  # JSONs are easier to work with in Python
        keep_alive=60,  # How long the model should be kept in memory after the call
    )

    return json.loads(responce["message"]["content"])
