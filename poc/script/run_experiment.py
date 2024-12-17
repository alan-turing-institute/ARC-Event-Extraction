from __future__ import annotations

import csv
import os
from typing import Any

import pandas as pd
from numpy.random import RandomState
from ollama import Client

from poc.prompt import eall_prompt, etrig_prompt, etype_prompt
from poc.query import (
    get_ollama_responce,
    hand_close_examples,
    hand_no_examples,
    hand_random_examples,
)

DATA_PATH = "data/gpt_data_with_embeddings.json"
RESULTS_PATH = "result/e_all_close_experiment.csv"
RANDOM_SEED = 1337
EXAMPLE_TYPE = "close_example"
TOTAL_SAMPLES = 100
TASK_TYPE = "event_all"


def write_event_type_results(
    writer: csv.writer, outputs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> int:
    writer.writerow(["Index", "RealEventType", "GivenEventType", "IsMatch"])

    total_correct = 0
    for output in outputs:
        given_event = output[1].get("EventType", "")

        real_event = output[0]["EventType"]
        is_match = real_event == given_event

        if is_match:
            total_correct += 1

        writer.writerow([output[0]["Index"], real_event, given_event, is_match])

    return total_correct


def write_event_trigger_results(
    writer: csv.writer, outputs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> int:
    writer.writerow(
        [
            "Index",
            "RealEventType",
            "RealEventTrigger",
            "GivenEventTrigger",
            "GivenEventTrigger",
            "IsMatch",
        ]
    )

    total_correct = 0

    for output in outputs:
        given_event = output[1].get("EventType", "")
        given_trigger = output[1].get("EventTrigger", "")

        real_event = output[0]["EventType"]
        real_trigger = output[0]["EventTrigger"]
        is_match = real_event == given_event and real_trigger == given_trigger

        if is_match:
            total_correct += 1

        writer.writerow(
            [
                output[0]["Index"],
                real_event,
                real_trigger,
                given_event,
                given_trigger,
                is_match,
            ]
        )

    return total_correct


def write_event_all_results(
    writer: csv.writer, outputs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> int:
    writer.writerow(
        [
            "Index",
            "RealEventType",
            "RealEventTrigger",
            "RealSubject",
            "RealObject",
            "GivenEventTrigger",
            "GivenEventTrigger",
            "GivenSubject",
            "GivenObject",
            "IsMatch",
        ]
    )

    total_correct = 0

    for output in outputs:
        given_event = output[1].get("EventType", "")
        given_trigger = output[1].get("EventTrigger", "")
        given_subject = output[1].get("Subject", "")
        given_object = output[1].get("Object", "")

        real_event = output[0]["EventType"]
        real_trigger = output[0]["EventTrigger"]
        real_subject = output[0]["Subject"]
        real_object = output[0]["Object"]
        is_match = (
            real_event == given_event
            and real_trigger == given_trigger
            and real_subject == given_subject
            and real_object == given_object
        )

        if is_match:
            total_correct += 1

        writer.writerow(
            [
                output[0]["Index"],
                real_event,
                real_trigger,
                real_subject,
                real_object,
                given_event,
                given_trigger,
                given_subject,
                given_object,
                is_match,
            ]
        )

    return total_correct


def main() -> None:
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_df = pd.read_json(
        os.path.join(main_dir, DATA_PATH), orient="records", lines=True
    )

    random_state = RandomState(RANDOM_SEED)
    sample_choice = random_state.choice(data_df.index, TOTAL_SAMPLES, replace=False)

    ollama_client = Client()

    if TASK_TYPE == "event_type":
        prompt_func = etype_prompt
    elif TASK_TYPE == "event_trigger":
        prompt_func = etrig_prompt
    elif TASK_TYPE == "event_all":
        prompt_func = eall_prompt
    else:
        msg = f"Invalid TASK_TYPE: {TASK_TYPE}"
        raise ValueError(msg)

    def query_ollama(sent, examps):
        returned_prompt = prompt_func(sent, examples=examps)
        return get_ollama_responce(ollama_client, "llama3.2:3b", returned_prompt)

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

        if TASK_TYPE == "event_type":
            correct = write_event_type_results(writer, outputs)
        elif TASK_TYPE == "event_trigger":
            correct = write_event_trigger_results(writer, outputs)
        elif TASK_TYPE == "event_all":
            correct = write_event_all_results(writer, outputs)
        else:
            msg = f"Invalid TASK_TYPE: {TASK_TYPE}"
            raise ValueError(msg)

    print(f"Total correct: {correct}/{TOTAL_SAMPLES}")


if __name__ == "__main__":
    main()
