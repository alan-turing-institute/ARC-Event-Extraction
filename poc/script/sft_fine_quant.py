from __future__ import annotations

import os

import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

wandb.login()

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_LOC = os.path.join(main_dir, "model/Llama-3.2-3B-Instruct")
MADE_MODEL_LOC = os.path.join(main_dir, "model/ADT4b-Llama-3.2-3B-Instruct")
TRAIN_DATA = os.path.join(main_dir, "data/gpt_train.tsv")
EVAL_DATA = os.path.join(main_dir, "data/gpt_eval.tsv")

CP_MODEL_PATH = os.path.join(main_dir, "model/output/checkpoint-180")

LOG_DIR = os.path.join(main_dir, "logs")
OUTPUT_DIR = os.path.join(main_dir, "model/output")

QUANTISE = True


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
    answer = (
        f"{{'EventType': '{example['EventType']}', "
        f"'EventTrigger': '{example['EventTrigger']}', "
        f"'Subject': '{example['Subject']}', "
        f"'Object': '{example['Object']}'}}"
    )
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True) if QUANTISE else None

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOC, quantization_config=bnb_config
    )

    # Load the dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": TRAIN_DATA, "eval": EVAL_DATA},
        delimiter="\t",
    )

    formatted_dataset = dataset.map(
        format_chat_message, remove_columns=dataset["train"].column_names
    )

    # And the completion only settings
    responce_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(responce_template, tokenizer=tokenizer)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Config Settings
    training_args = SFTConfig(
        # packing=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        report_to="wandb",
        log_level="info",
        logging_steps=30,
        do_eval=True,
        eval_strategy="steps",
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        weight_decay=0.1,
        # neftune_noise_alpha=1,
    )
    peft_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        use_rslora=True,
        bias="none",
        # init_lora_weights="pissa",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "v_proj", "up_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "up_proj",
        #   "down_proj", "o_proj", "gate_proj"],
    )

    # Train the model
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["eval"],
        peft_config=peft_config,
        data_collator=collator,
    )

    trainer.train()

    # Save the model
    trainer.model.save_pretrained(MADE_MODEL_LOC)


if __name__ == "__main__":
    main()
