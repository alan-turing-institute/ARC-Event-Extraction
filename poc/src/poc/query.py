from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from ollama import Client
from tqdm import tqdm

__all__ = [
    "get_ollama_responce",
    "hand_no_examples",
    "hand_random_examples",
    "hand_close_examples",
]


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


def hand_no_examples(
    data_df: pd.DataFrame,
    sample_choice: list[int],
    prompt_func: Callable[[str], dict[str, Any]],
    verbose: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Processes a selection of samples from a DataFrame and applies a prompt function to
    each sample. This function does pass any examples of correct sentences to the
    prompt function.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the sentence samples.
        sample_choice (list[int]): A list of indices indicating which samples to
            process.
        prompt_func (Callable): A function that takes a sentence (str) and returns a
            processed result.
        verbose (bool, optional): If True, displays a progress bar. Defaults to False.

    Returns:
        list[tuple[dict]]: A list of tuples, each containing the original sample
            (as a dictionary) and the result of the prompt function.
    """
    samples = data_df[data_df["Index"].isin(sample_choice)].to_dict("records")
    output = []
    for samp in tqdm(samples, disable=not verbose):
        output.append((samp, prompt_func(samp["Sentence"])))
    return output


def hand_random_examples(
    data_df: pd.DataFrame,
    sample_choice: list[int],
    prompt_func: Callable[[str, list[dict]], dict[str, Any]],
    number_samples: int = 5,
    random_state: np.RandomState | None = None,
    verbose: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Processes a selection of samples from a DataFrame and applies a prompt function to
    each sample. This function chooses a random selection of examples to pass to the
    prompt function. This random selection is made

    Args:
        data_df (pd.DataFrame): The DataFrame containing the data.
        sample_choice (list[int]): List of indices to select specific samples from the
            DataFrame.
        prompt_func (Callable[[str, list[dict]], dict]): A function that takes a
            sentence and a list of examples, and returns a dictionary.
        number_samples (int, optional): The number of random samples to select for each
            chosen sample. Defaults to 5.
        random_state (np.RandomState, optional): The random state to use.
        verbose (bool, optional): If True, displays a progress bar. Defaults to False.

    Returns:
        list[tuple[dict]]: A list of tuples, each containing a sample and the result of
            applying the prompt function to that sample.
    """
    samples = data_df[data_df["Index"].isin(sample_choice)].to_dict("records")
    output = []
    for samp in tqdm(samples, disable=not verbose):
        examples = (
            data_df[data_df["Index"] != samp["Index"]]
            .sample(number_samples, random_state=random_state)
            .to_dict("records")
        )
        output.append((samp, prompt_func(samp["Sentence"], examples)))

    return output


def cosine_similarity(list1: list[float], list2: list[float]) -> float:
    """
    Calculate the cosine similarity between two lists of floats.

    Args:
        list1 (list[float]): The first list of floats.
        list2 (list[float]): The second list of floats.

    Returns:
        float: The cosine similarity between the two lists.
    """
    vec1 = np.array(list1)
    vec2 = np.array(list2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)


def hand_close_examples(
    data_df: pd.DataFrame,
    sample_choice: list[int],
    prompt_func: Callable[[str, list[dict]], dict[str, Any]],
    number_samples: int = 5,
    verbose: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Processes a selection of samples from a DataFrame and applies a prompt function to
    each sample. This function selects the closest samples to each chosen sample based
    on cosine similarity between the embeddings of the samples.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the data.
        sample_choice (list[int]): The indices of the samples to process.
        prompt_func (Callable[[str, list[dict]], dict]): A function that takes a
            sentence and a list of examples, and returns a dictionary.
        number_samples (int, optional): The number of closest samples to select for
            each chosen sample. Defaults to 5.
        verbose (bool, optional): If True, displays a progress bar. Defaults to False.

    Returns:
        list[tuple[dict]]: A list of tuples, each containing a sample and the result of
            applying the prompt function to that sample.
    """
    samples = data_df[data_df["Index"].isin(sample_choice)].to_dict("records")
    output = []
    for samp in tqdm(samples, disable=not verbose):
        closeness = data_df[data_df["Index"] != samp["Index"]]["Embedding"].map(
            lambda x, sent=samp["Embedding"]: cosine_similarity(x, sent)
        )
        examples = closeness.nlargest(number_samples).index.map(
            lambda x: data_df.loc[x].to_dict()
        )
        output.append((samp, prompt_func(samp["Sentence"], examples)))

    return output
