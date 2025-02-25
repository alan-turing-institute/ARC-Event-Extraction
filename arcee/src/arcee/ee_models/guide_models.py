from __future__ import annotations

import os

import guidance as gi
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcee.ee_models.ee_models import EEBaseModel


class Llama3ChatTemplate(gi.chat.ChatTemplate):
    # available_roles = ["user", "assistant"]
    # template_str = llama31_template

    def get_role_start(self, role_name: str) -> str:
        if role_name == "system":
            return "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful!"
        if role_name == "user":
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        if role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        raise gi.chat.UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name: str | None = None) -> str:
        del role_name  # Unused include for compatibility
        return "<|eot_id|>"


CHAT_TEMPLATES = {
    "llama3": Llama3ChatTemplate,
    "llama2": gi.chat.Llama2ChatTemplate,
    "Phi3Mini": gi.chat.Phi3MiniChatTemplate,
}


class BaseGuideModel(EEBaseModel):
    """
    This uses the a Hugging Face model and a Guidence package to contrain a model so
    it gives JSON.

    Attributes:
        model: An instance of a `transformers.AutoModelForCausalLM`. Should be an
            instruction-tuned model whose tokenizer has a chat template.
        tokenizer: An instance of a `transfomers.AutoTokenizer`, corresponding to the
            model. Should contain a chat template.
        device: A string denoting what device to place the model on, defaults to `cpu`.
            Pass `device_map="auto"` to init to automatically place on any available
            GPU.
        options: Currently ignored, kept in to match base class signature.
    """

    def __init__(
        self,
        model_id: str,
        tokenizer_path: str | None = None,
        chat_template: str | None = None,
        options: dict | None = None,
    ) -> None:
        # Options - not needed in basic implementation
        self.options = options

        # Need to set this environment variable to use guidance
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_id
        hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side="left"
        )

        # Setup special tokens for batching
        if hf_tokenizer.pad_token_id is None:
            hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id

        # Setup chat template
        if chat_template not in CHAT_TEMPLATES:
            msg = (
                f"Chat template {chat_template} not found in "
                f"{' '.join(list(CHAT_TEMPLATES.keys()))}"
            )
            raise ValueError(msg)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Load model
        self.device = device
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
        )
        self.hf_model.eval()

        # Wrap in guidance model
        self.model = gi.models.Transformers(
            model=self.hf_model,
            tokenizer=hf_tokenizer,
            chat_template=CHAT_TEMPLATES[chat_template],
        )


class GuideFreeJSONModel(BaseGuideModel):
    """
    This uses the a Hugging Face model and a Guidence package to contrain a model so
    it gives JSON.

    Attributes:
        model: An instance of a `transformers.AutoModelForCausalLM`. Should be an
            instruction-tuned model whose tokenizer has a chat template.
        tokenizer: An instance of a `transfomers.AutoTokenizer`, corresponding to the
            model. Should contain a chat template.
        device: A string denoting what device to place the model on, defaults to `cpu`.
            Pass `device_map="auto"` to init to automatically place on any available
            GPU.
        options: Currently ignored, kept in to match base class signature.
    """

    def __init__(
        self,
        model_id: str,
        tokenizer_path: str | None = None,
        chat_template: str | None = None,
        max_tokens: int = 500,
        options: dict | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            tokenizer_path=tokenizer_path,
            chat_template=chat_template,
            options=options,
        )
        self.max_tokens = max_tokens

    def predict(self, input: list[dict[str, str]]) -> list[str]:
        """This uses the

        Args:
            input (list[dict[str, str]]): _description_

        Returns:
            list[str]: _description_
        """
        responses = []
        for chat in input:
            lm = self.model.copy()
            with gi.user():
                lm += chat["content"]
            with gi.assistant():
                lm += "```json\n\n"
                lm += gi.json(name="output", max_tokens=self.max_tokens)
                lm += "\n\n```"

            responses.append(lm["output"])
        return responses
