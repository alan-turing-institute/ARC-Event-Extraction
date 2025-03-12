from __future__ import annotations

import os

import wandb
from arcee.dataset.load_training_eval import load_training_eval_data
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

wandb.login()

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_LOC = os.path.join(main_dir, "models/Llama-3.2-3B-Instruct")
MADE_MODEL_LOC = os.path.join(main_dir, "models/ADT-eae-Llama-3.2-3B-Instruct")

TRAIN_DATA = "data/ACE05/output/split_dev/train.jsonlines"
EVAL_DATA = "data/ACE05/output/split_dev/dev.jsonlines"
PROMPT_CLS = "EAEPrompter1"
SCHEMA_PATH = "data/ACE05/schema/pattern.json"

LOG_DIR = os.path.join(main_dir, "logs")
OUTPUT_DIR = os.path.join(main_dir, "models/output")


def main():
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC)
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOC)

    # Load the dataset
    dataset = load_training_eval_data(TRAIN_DATA, EVAL_DATA, PROMPT_CLS, SCHEMA_PATH)

    # And the completion only settings
    responce_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(responce_template, tokenizer=tokenizer)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Config Settings
    training_args = SFTConfig(
        # packing=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        report_to="wandb",
        log_level="info",
        logging_steps=60,
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
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        data_collator=collator,
    )

    trainer.train()

    # Save the model
    trainer.model.save_pretrained(MADE_MODEL_LOC)


if __name__ == "__main__":
    main()
