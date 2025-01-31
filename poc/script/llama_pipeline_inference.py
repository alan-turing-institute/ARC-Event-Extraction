from __future__ import annotations

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_LOC = os.path.join(main_dir, "model/Llama-3.2-1B-Instruct")
TRAIN_DATA = os.path.join(main_dir, "data/gpt_train.tsv")
EVAL_DATA = os.path.join(main_dir, "data/gpt_eval.tsv")
TEST_DATA = os.path.join(main_dir, "data/gpt_test.tsv")

RESULTS_DIR = os.path.join(main_dir, "results/llama1b.txt")


def format_chat_message(example):
    """This function formats the chat message for the user.

    Args:
        example (dict): A dictionary containing the sentence, event type, event
            trigger, subject, and object.

    Returns:
        dict: A dictionary containing the formatted chat message.
    """
    event_types = (
        "Build",
        "Clean",
        "Cook",
        "Exercise",
        "Hire",
        "Injure",
        "Meet",
        "Purchase",
        "Transport",
        "Travel",
    )
    question = (
        f"I am going to give you a sentence and I want you to tell me what type "
        "of event it is describing and what is the indicating verb, object, "
        "and subject of the sentence. There are ten possible types of events "
        "you can choose from. "
        "Please give your answer in a JSON with the key 'EventType' matching "
        f"one of the following options: {event_types}. The 'EventTrigger' "
        "matching the relevant verb in the sentence. The 'Subject' matching "
        "the actor in the sentence, and 'Object' matching the target object.\n"
        f"The sentence is: {example['Sentence']}"
    )
    # answer = (
    #     f"{{'EventType': '{example['EventType']}', "
    #     f"'EventTrigger': '{example['EventTrigger']}', "
    #     f"'Subject': '{example['Subject']}', "
    #     f"'Object': '{example['Object']}'}}"
    # )
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            # {"role": "assistant", "content": answer},
        ]
    }


def main():
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOC)

    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": TRAIN_DATA, "eval": EVAL_DATA},
        delimiter="\t",
    )

    formatted_dataset = dataset.map(
        format_chat_message, remove_columns=dataset["train"].column_names
    )

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    outputs = pipe(formatted_dataset["train"]["messages"], batch_size=8)

    with open(RESULTS_DIR, "w") as f:
        for output in outputs:
            for message in output:
                f.write(str(message))
                f.write("\n")
            f.write("\n")


if __name__ == "__main__":
    main()
