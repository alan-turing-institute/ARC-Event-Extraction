from __future__ import annotations

import csv
import os

import pandas as pd
from numpy.random import RandomState
from ollama import Client

from poc.prompt import et_prompt
from poc.query import (
    get_ollama_responce,
    hand_close_examples,
    hand_no_examples,
    hand_random_examples,
)

DATA_PATH = "data/gpt_data_with_embeddings.json"
RESULTS_PATH = "result/et_close_experiment.csv"
RANDOM_SEED = 1337
EXAMPLE_TYPE = "close_example"
TOTAL_SAMPLES = 100


def main() -> None:
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_df = pd.read_json(
        os.path.join(main_dir, DATA_PATH), orient="records", lines=True
    )

    random_state = RandomState(RANDOM_SEED)
    sample_choice = random_state.choice(data_df.index, TOTAL_SAMPLES, replace=False)

    ollama_client = Client()

    def query_ollama(sent, examps):
        prompt = et_prompt(sent, examples=examps)
        return get_ollama_responce(ollama_client, "llama3.2:3b", prompt)

    if EXAMPLE_TYPE == "no_example":
        hand_example = hand_no_examples
    elif EXAMPLE_TYPE == "random_example":
        hand_example = hand_random_examples  # need to hand seed
    elif EXAMPLE_TYPE == "close_example":
        hand_example = hand_close_examples  # need to hand example number
    else:
        msg = f"Invalid EXAMPLE_TYPE: {EXAMPLE_TYPE}"
        raise ValueError(msg)

    outputs = hand_example(data_df, sample_choice, query_ollama, verbose=True)

    with open(os.path.join(main_dir, RESULTS_PATH), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "RealEventType", "GivenEventType"])

        for output in outputs:
            given_event = output[0].get("EventType", "")

            real_event = output[0]["EventType"]
            is_match = real_event == given_event

            writer.writerow([output[0]["Index"], real_event, given_event, is_match])


if __name__ == "__main__":
    main()
