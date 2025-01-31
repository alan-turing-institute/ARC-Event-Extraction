from __future__ import annotations

import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_LOC = os.path.join(main_dir, "model/Llama-3.2-1B-Instruct")
TRAIN_DATA = os.path.join(main_dir, "data/gpt_train.tsv")
EVAL_DATA = os.path.join(main_dir, "data/gpt_eval.tsv")
TEST_DATA = os.path.join(main_dir, "data/gpt_test.tsv")

RESULTS_DIR = os.path.join(main_dir, "results/batch_llama1b.txt")


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
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
        # {"role": "assistant", "content": answer},
    ]


messages = []

darta_df = pd.read_csv(TEST_DATA, sep="\t")

for _idx, row in darta_df.iterrows():
    messages.append(format_chat_message(row))

tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL_LOC, torch_dtype=torch.bfloat16, device_map="auto"
)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


texts = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
inputs = tokenizer(texts, padding="longest", return_tensors="pt")
temp_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

n_batches = len(inputs["input_ids"]) // 16 + 1
all_gen_text = []
model.eval()
model.to("mps")

for i in tqdm(range(n_batches)):
    start = i * 16
    end = (i + 1) * 16
    batch_inputs = {key: val[start:end].to("mps") for key, val in inputs.items()}
    gen_tokens = model.generate(
        **batch_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    all_gen_text.extend(gen_text)
    del batch_inputs


all_gen_text = [i[len(temp_texts[idx]) :] for idx, i in enumerate(all_gen_text)]

with open(RESULTS_DIR, "w") as f:
    for _idx, (text, gen) in enumerate(zip(temp_texts, all_gen_text, strict=False)):
        f.write(f"Text: {text}\n")
        f.write(f"Generated: {gen}\n")
        f.write("\n")
