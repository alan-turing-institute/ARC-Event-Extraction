from __future__ import annotations

import json
import os

import guidance as gi
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        if "type" in given_json and given_json["type"] == "object":
            given_json["additionalProperties"] = False
        for value in given_json.values():
            remove_unimplemented_keys(value)
    elif isinstance(given_json, list):
        for item in given_json:
            remove_unimplemented_keys(item)

    return given_json


def load_quatised_config(quant_config: dict | None) -> BitsAndBytesConfig:
    """
    Load a quantised config from a dictionary.
    If no config is given then the default is to load in 4bit.

    This function will raise an error if the device is not "cuda" as quantisation
    is only available on CUDA devices. It will also raise an error if the optional
    dependencies are not installed.

    Args:
        quant_config (dict, optional): The quantisation config to load. If not given
            then the default is to load in 4bit. Defaults to None.

    Returns:
        BitsAndBytesConfig: The quantised config.
    """
    if not torch.cuda.is_available():
        msg = "Quantisation only available on CUDA devices."
        raise ValueError(msg)

    try:
        import bitsandbytes  # This checks if the optional dependencies are installed

        del bitsandbytes
    except ImportError as exc:
        msg = "Need the optional 'cuda' dependencies to use quantisation."
        raise ImportError(msg) from exc

    if quant_config:
        bnb_config = BitsAndBytesConfig(**quant_config)
    else:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    return bnb_config


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


class MistralChatTemplate(gi.chat.ChatTemplate):
    def get_role_start(self, role_name: str) -> str:
        if role_name == "user":
            return " [INST] "
        if role_name == "assistant":
            return " "
        if role_name == "system":
            return " [SYSTEM_PROMPT] "
        raise gi.chat.UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name: str | None = None) -> str:
        if role_name == "user":
            return " [/INST]"
        if role_name == "assistant":
            return "</s>"
        if role_name == "system":
            return " [/SYSTEM_PROMPT]"
        raise gi.chat.UnsupportedRoleException(role_name, self)


CHAT_TEMPLATES = {
    "llama3": Llama3ChatTemplate,
    "mistral": MistralChatTemplate,
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
        adapter_id: str | None = None,
        chat_template: str = "llama3",
        quantised: bool = False,
        quant_config: dict | None = None,
        options: dict | None = None,
    ) -> None:
        """
        Initialise the base model which will use constrained generation.

        Args:
            model_id (str): The location of the HF model to load.
            adapter_id (str | None, optional): The location of the adapter to
                load if different from the base model. Defaults to None.
            chat_template (str): The chat template to use with the model, this
                should be one of the keys in `CHAT_TEMPLATES`. Defaults to "llama3".
            quantised (bool, optional): If the model is quantised. Defaults to False.
            quant_config (dict | None, optional): The quantisation config to use.
                Defaults to None which will load in 4bit.
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
        hf_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(models_dir, model_id), padding_side="left"
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

        bnb_config = load_quatised_config(quant_config) if quantised else None

        base_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(models_dir, model_id),
            quantization_config=bnb_config,
        )

        # Load Model
        if adapter_id:
            self.hf_model = PeftModel.from_pretrained(
                base_model, os.path.join(models_dir, adapter_id)
            )
        else:
            self.hf_model = base_model

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
        adapter_id: The location of the adapted file weights to update the model.


        options: Currently ignored, kept in to match base class signature.
    """

    def __init__(
        self,
        model_id: str,
        adapter_id: str | None = None,
        chat_template: str = "llama3",
        quantised: bool = False,
        quant_config: dict | None = None,
        max_tokens: int = 500,
        options: dict | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            adapter_id=adapter_id,
            chat_template=chat_template,
            quantised=quantised,
            quant_config=quant_config,
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
                lm += "```json\n"
                lm += gi.gen(name="output", stop="\n```", max_tokens=self.max_tokens)
                lm += "\n```"

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
        whitespace_flexible: If whitespace is allowed in the JSON.
    """

    def __init__(
        self,
        model_id: str,
        adapter_id: str | None = None,
        chat_template: str = "llama3",
        quantised: bool = False,
        quant_config: dict | None = None,
        options: dict | None = None,
        max_tokens: int = 500,
        whitespace_flexible: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            adapter_id=adapter_id,
            chat_template=chat_template,
            quantised=quantised,
            quant_config=quant_config,
            options=options,
        )
        self.max_tokens = max_tokens
        self.whitespace_flexible = whitespace_flexible

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
                lm += "```json\n"
                lm += gi.json(
                    name="output",
                    schema=True,
                    max_tokens=self.max_tokens,
                    whitespace_flexible=self.whitespace_flexible,
                )
                lm += "\n```"

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
        whitespace_flexible: If whitespace is allowed in the JSON.
    """

    def __init__(
        self,
        model_id: str,
        adapter_id: str | None = None,
        chat_template: str = "llama3",
        quantised: bool = False,
        quant_config: dict | None = None,
        options: dict | None = None,
        max_tokens: int = 500,
        whitespace_flexible: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            adapter_id=adapter_id,
            chat_template=chat_template,
            quantised=quantised,
            quant_config=quant_config,
            options=options,
        )
        self.max_tokens = max_tokens
        self.whitespace_flexible = whitespace_flexible

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
                lm += "```json\n"
                lm += gi.json(
                    name="output",
                    schema=striped_schema,
                    max_tokens=self.max_tokens,
                    whitespace_flexible=self.whitespace_flexible,
                )
                lm += "\n```"

            try:
                output = json.loads(lm["output"])
            except json.JSONDecodeError:
                output = {"JSON_ERROR": lm["output"]}

            response = {**ids, "output": output}
            responses.append(response)

        return responses


class GuideOneEventModel(BaseGuideModel):
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
        whitespace_flexible: If whitespace is allowed in the JSON.
    """

    def __init__(
        self,
        model_id: str,
        adapter_id: str | None = None,
        chat_template: str = "llama3",
        quantised: bool = False,
        quant_config: dict | None = None,
        options: dict | None = None,
        max_tokens: int = 500,
        whitespace_flexible: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            adapter_id=adapter_id,
            chat_template=chat_template,
            quantised=quantised,
            quant_config=quant_config,
            options=options,
        )
        self.max_tokens = max_tokens
        self.whitespace_flexible = whitespace_flexible

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
        for chat, schema, ids, sentence in input:
            if isinstance(schema, str):
                msg = "Schema should be a dictionary not a string."
                raise ValueError(msg)
            key_list = list(schema["items"]["properties"].keys())
            lm = self.model.copy()
            with gi.user():
                lm += chat["content"]
            with gi.assistant():
                lm += "```json\n["
                lm += gi.select(["]\n", "\n  {\n"], name="if_output")
                if lm["if_output"] == "\n  {\n":
                    lm += '   "' + gi.select(key_list, name="key")
                    lm += '": "' + gi.substring(sentence, name="value") + '"'
                    lm += "\n  }\n"
                lm += "```"

            output: list[dict] | dict = []
            try:
                output = [] if lm["if_output"] == "]\n" else [{lm["key"]: lm["value"]}]
            except json.JSONDecodeError:
                output = {"JSON_ERROR": lm["output"]}

            response = {**ids, "output": output}
            responses.append(response)

        return responses
