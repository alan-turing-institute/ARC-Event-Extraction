from __future__ import annotations

import json
import os

import guidance as gi
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcee.ee_models.ee_models import EEBaseModel


def remove_unimplemented_keys(
    given_json: str | list | dict | None,
) -> str | list | dict | None:
    """
    Remove unimplemented keys from a JSON schema as guidance does not support
    `minProperties` and `maxProperties`.

    Args:
        given_json (list | dict): The JSON schema to remove unimplemented keys from.

    Returns:
        list | dict: The JSON schema with unimplemented keys removed.
    """
    if isinstance(given_json, dict):
        given_json.pop("maxProperties", None)
        given_json.pop("minProperties", None)
        for value in given_json.values():
            remove_unimplemented_keys(value)
    elif isinstance(given_json, list):
        for item in given_json:
            remove_unimplemented_keys(item)

    return given_json


class Llama3ChatTemplate(gi.chat.ChatTemplate):
    # available_roles = ["user", "assistant"]
    # template_str = llama31_template

    def get_role_start(self, role_name: str) -> str:
        """
        This method returns the string which precedes the content of the message.
        It needs to be consistent with how the model was trained.
        """
        if role_name == "system":
            return "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful!"
        if role_name == "user":
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        if role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        raise gi.chat.UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name: str | None = None) -> str:
        """
        This method returns the string which follows the content of the message.
        """
        del role_name  # Unused include for compatibility
        return "<|eot_id|>"


CHAT_TEMPLATES = {
    "llama3": Llama3ChatTemplate,
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
        device: A string denoting what device to place the model on.
        options: Currently ignored, kept in to match base class signature.
    """

    def __init__(
        self,
        model_id: str,
        tokenizer_path: str | None = None,
        chat_template: str = "llama3",
        options: dict | None = None,
    ) -> None:
        """
        Initialise the base model which will use constrained generation.

        Args:
            model_id (str): The location of the HF model to load.
            tokenizer_path (str | None, optional): The location of the tokenizer to
                load if different from the model. Defaults to None.
            chat_template (str): The chat template to use with the model, this
                should be one of the keys in `CHAT_TEMPLATES`. Defaults to "llama3".
            options (dict | None, optional): Currently ignored might be needed for
                classes that inherit from this. Defaults to None.
        """
        # Options - not needed in basic implementation
        self.options = options

        # Need to set this environment variable to use guidance
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        models_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            ),
            "models",
        )

        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_id
        hf_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(models_dir, tokenizer_path), padding_side="left"
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
            os.path.join(models_dir, model_id),
            device_map=self.device,
        )
        self.hf_model.eval()

        # Wrap in guidance model
        self.model = gi.models.Transformers(
            model=self.hf_model,
            tokenizer=hf_tokenizer,
            chat_template=CHAT_TEMPLATES[chat_template],
        )


class GuideFreeModel(BaseGuideModel):
    """
    This uses the a Hugging Face model and a Guidence package but does not constrain
    the model.

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
        chat_template: str = "llama3",
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

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
    ) -> list[dict]:
        """
        This method takes a list of inputs and returns a list of outputs. Each input
        is a event or passage to run the event extraction model on. The output is a
        list of events extracted from the input.

        Args:
            input (list[tuple]): A list of tuples where each tuple requires extraction.
                - prompt (dict[str, str]): The chat prompt to use.
                - schema (str | dict): The schema to use or 'json' (ignored).
                - ids (dict): The ids to use.
                - sentence (str): The sentence to extract events from (ignored).

        Returns:
            list: A list of dictionaries where each dictionary contains the extracted
                events in the schema. If the output cannot be parsed as JSON then the
                output is a dictionary with a key 'JSON_ERROR' and the value is the
                output.
        """
        responses = []
        for chat, _, ids, _ in input:
            lm = self.model.copy()
            with gi.user():
                lm += chat["content"]
            with gi.assistant():
                lm += "```json\n\n"
                lm += gi.gen(name="output", stop="\n\n```", max_tokens=self.max_tokens)
                lm += "\n\n```"

            try:
                output = json.loads(lm["output"])
            except json.JSONDecodeError:
                output = {"JSON_ERROR": lm["output"]}

            response = {**ids, "output": output}
            responses.append(response)

        return responses


class GuideJSONModel(BaseGuideModel):
    """
    This uses the a Hugging Face model and a Guidence package to contrain a model so
    it gives a valid JSON.

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
        chat_template: str = "llama3",
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

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
    ) -> list[dict]:
        """
        This method takes a list of inputs and returns a list of outputs. Each input
        is a event or passage to run the event extraction model on. The output is a
        list of events extracted from the input.

        Args:
            input (list[tuple]): A list of tuples where each tuple requires extraction.
                - prompt (dict[str, str]): The chat prompt to use.
                - schema (str | dict): The schema to use or 'json' (ignored).
                - ids (dict): The ids to use.
                - sentence (str): The sentence to extract events from  (ignored).

        Returns:
            list: A list of dictionaries where each dictionary contains the extracted
                events in the schema. If the output cannot be parsed as JSON then the
                output is a dictionary with a key 'JSON_ERROR' and the value is the
                output.
        """
        responses = []
        for chat, _, ids, _ in input:
            lm = self.model.copy()
            with gi.user():
                lm += chat["content"]
            with gi.assistant():
                lm += "```json\n\n"
                lm += gi.json(name="output", schema="json", max_tokens=self.max_tokens)
                lm += "\n\n```"

            try:
                output = json.loads(lm["output"])
            except json.JSONDecodeError:
                output = {"JSON_ERROR": lm["output"]}

            response = {**ids, "output": output}
            responses.append(response)

        return responses


class GuideSchemaModel(BaseGuideModel):
    """
    This uses the a Hugging Face model and a Guidence package to contrain a model so
    it gives JSON in a schema.

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
        chat_template: str = "llama3",
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

    def predict(
        self, input: list[tuple[dict[str, str], str | dict, dict, str]]
    ) -> list[dict]:
        """
        This method takes a list of inputs and returns a list of outputs. Each input
        is a event or passage to run the event extraction model on. The output is a
        list of events extracted from the input.

        Args:
            input (list[tuple]): A list of tuples where each tuple requires extraction.
                - prompt (dict[str, str]): The chat prompt to use.
                - schema (str | dict): The schema to use or 'json' (ignored).
                - ids (dict): The ids to use.
                - sentence (str): The sentence to extract events from (ignored).

        Returns:
            list: A list of dictionaries where each dictionary contains the extracted
                events in the schema. If the output cannot be parsed as JSON then the
                output is a dictionary with a key 'JSON_ERROR' and the value is the
                output.
        """
        responses = []
        for chat, schema, ids, _ in input:
            striped_schema = remove_unimplemented_keys(schema)
            lm = self.model.copy()
            with gi.user():
                lm += chat["content"]
            with gi.assistant():
                lm += "```json\n\n"
                lm += gi.json(
                    name="output", schema=striped_schema, max_tokens=self.max_tokens
                )
                lm += "\n\n```"

            try:
                output = json.loads(lm["output"])
            except json.JSONDecodeError:
                output = {"JSON_ERROR": lm["output"]}

            response = {**ids, "output": output}
            responses.append(response)

        return responses
