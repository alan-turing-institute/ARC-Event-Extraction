from __future__ import annotations

import argparse
import os

import wandb
import yaml
from arcee.constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from arcee.dataset.load_training_eval import load_training_eval_data
from arcee.ee_models.guide_models import load_quatised_config
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

wandb.login()

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_loc: str) -> dict:
    """Takes a config file and returns the infotmation."""
    with open(os.path.join(main_dir, config_loc)) as file:
        return yaml.safe_load(file)


def main(config_loc: str):
    config = load_config(config_loc)

    # Initialise wandb run
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        name=f"fine_tune_{config['adapter_loc']}",
        config={
            "job_type": "Fine Tuning",
            **config,
        },
    )

    # Quantised Training

    if config.get("quantised"):
        q_config = config.get("quant_config", None)
        bnb_config = load_quatised_config(q_config)
    else:
        bnb_config = None

    # Load the model
    model_loc = os.path.join(main_dir, config["model_loc"])
    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    model = AutoModelForCausalLM.from_pretrained(
        model_loc, quantization_config=bnb_config
    )

    # Load the dataset
    train_data = os.path.join(main_dir, config["train_data"])
    eval_data = os.path.join(main_dir, config["eval_data"])
    schema_path = os.path.join(main_dir, config["schema_path"])
    dataset = load_training_eval_data(
        train_data,
        eval_data,
        config["prompt_cls"],
        schema_path,
        config.get("only_events"),
        config.get("null_event_ratio", 0),
        config.get("null_event_seed", 0),
    )

    model_type = config.get("model_type")

    if not model_type or model_type == "llama":
        responce_template = "<|start_header_id|>assistant<|end_header_id|>"
        collator = DataCollatorForCompletionOnlyLM(
            responce_template, tokenizer=tokenizer
        )
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif model_type == "mistral":
        responce_template = "[/INST]"
        collator = DataCollatorForCompletionOnlyLM(
            responce_template, tokenizer=tokenizer
        )
        tokenizer.pad_token = "<pad>"
    else:
        msg = "model_type must be either llama or mistral"
        raise ValueError(msg)

    # Config Settings
    output_loc = config["stf_config"].get("output_dir")
    if output_loc:
        config["stf_config"]["output_dir"] = os.path.join(main_dir, output_loc)

    training_args = SFTConfig(**config["stf_config"])
    peft_config = LoraConfig(**config["peft_config"])

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
    adapter_loc = os.path.join(main_dir, config["adapter_loc"])
    trainer.model.save_pretrained(adapter_loc)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for event extraction."
    )
    parser.add_argument("config_loc", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config_loc)
