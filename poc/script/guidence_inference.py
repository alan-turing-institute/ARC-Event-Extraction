from __future__ import annotations

import csv
import os
from typing import Any

import guidance as gi
import pandas as pd
import torch
from numpy.random import RandomState
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from poc.guide import get_llm_responce, get_unguided_llm_responce
from poc.prompt import eall_prompt, etrig_prompt, etype_prompt
from poc.query import hand_close_examples, hand_no_examples, hand_random_examples

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_LOC = os.path.join(main_dir, "model/Llama-3.2-1B-Instruct")
MODEL_LOC = os.path.join(main_dir, "model/Llama-3.2-1B-Instruct")
DATA_PATH = "data/gpt_test.tsv"
RESULTS_PATH = "results/guide/eall_guide_close_llama1b.csv"
RANDOM_SEED = 1337
EXAMPLE_TYPE = "close_example"
TOTAL_SAMPLES = 100
TASK_TYPE = "event_all"
GUIDED = True
QUATISATION = None
SENT_TRANSFORMER_MODEL = "all-mpnet-base-v2"
EVENT_TYPE_LIST = [
    "Clean",
    "Cook",
    "Travel",
    "Exercise",
    "Meet",
    "Injure",
    "Purchase",
    "Transport",
    "Hire",
    "Build",
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class Llama3ChatTemplate(gi.chat.ChatTemplate):
    # available_roles = ["user", "assistant"]
    # template_str = llama31_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful!"
        if role_name == "user":
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        if role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        raise gi.chat.UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        del role_name  # Unused include for compatibility
        return "<|eot_id|>"


def load_llama3_model(model_loc: str, token_loc: str, quantisation: str | None):
    """Loads the Llama-3 model and tokenizer and wrapps it in a guidance model."""

    # Get GPU info
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if quantisation and device != "cuda":
        msg = "Quantisation only available on CUDA devices."
        raise ValueError(msg)

    if quantisation:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            msg = "Need the optional cuda dependencies to use quantisation."
            raise ImportError(msg) from exc

    # Pick the Quantisation config
    if quantisation == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantisation == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quant_config = None

    hf_tokenizer = AutoTokenizer.from_pretrained(
        token_loc, padding_side="left", trust_remote_code=True, local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_loc,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant_config,
    )
    if quantisation:
        hf_model.to(device)
    hf_model.eval()

    return gi.models.Transformers(
        model=hf_model,
        tokenizer=hf_tokenizer,
        chat_template=Llama3ChatTemplate,
    )


def main() -> None:
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_df = pd.read_csv(os.path.join(main_dir, DATA_PATH), sep="\t")
    data_df = data_df.reset_index().rename(columns={"index": "Index"})

    glmodel = load_llama3_model(MODEL_LOC, TOKEN_LOC, QUATISATION)

    if EXAMPLE_TYPE == "close_example":
        st_model = SentenceTransformer(SENT_TRANSFORMER_MODEL)
        data_df["Embedding"] = data_df["Sentence"].map(st_model.encode)

    random_state = RandomState(RANDOM_SEED)
    sample_choice = random_state.choice(data_df.index, TOTAL_SAMPLES, replace=False)

    if TASK_TYPE == "event_type":
        prompt_func = etype_prompt
    elif TASK_TYPE == "event_trigger":
        prompt_func = etrig_prompt
    elif TASK_TYPE == "event_all":
        prompt_func = eall_prompt
    else:
        msg = f"Invalid TASK_TYPE: {TASK_TYPE}"
        raise ValueError(msg)

    get_responce = get_llm_responce if GUIDED else get_unguided_llm_responce

    def query_llm(sent, examps):
        returned_prompt = prompt_func(sent, examples=examps)
        return get_responce(glmodel.copy(), returned_prompt, sent, EVENT_TYPE_LIST)

    if EXAMPLE_TYPE == "no_example":
        hand_example = hand_no_examples
    elif EXAMPLE_TYPE == "random_example":
        hand_example = hand_random_examples  # need to hand seed
    elif EXAMPLE_TYPE == "close_example":
        hand_example = hand_close_examples  # need to hand example number
    else:
        msg = f"Invalid EXAMPLE_TYPE: {EXAMPLE_TYPE}"
        raise ValueError(msg)

    outputs = hand_example(data_df, sample_choice, query_llm, verbose=True)

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
