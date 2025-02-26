from __future__ import annotations

from arcee.config.experiment import ExperimentConfig

EXPECTED_DATA_PATH = "data/RAMS/output/splits/split1/test.jsonlines"
EXPECTED_SCHEMA_PATH = "data/RAMS/patterns/pattern.json"
EXPECTED_ARG_EXTRACTION = False
EXPECTED_SHUFFLE = None
EXPECTED_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
EXPECTED_MODEL_TYPE = "hf"
EXPECTED_BATCH_SIZE = 8
EXPECTED_INIT_KWARGS = {
    "device_map": "cpu",
    "terminators": "<|eot_id|>",
}
EXPECTED_PROMPT_CLS = "DummyEAEPrompter"
EXPECTED_PROMPT_KWARGS = {}
EXPECTED_OUTPUT = "outputs/dummy_experiment.jsonl"


def test_DataArguments(dummy_experment_config: ExperimentConfig):
    assert dummy_experment_config.data_kwargs.data_path == EXPECTED_DATA_PATH
    assert dummy_experment_config.data_kwargs.schema_path == EXPECTED_SCHEMA_PATH
    assert dummy_experment_config.data_kwargs.shuffle is EXPECTED_SHUFFLE


def test_ModelArguments(dummy_experment_config: ExperimentConfig):
    assert dummy_experment_config.model_kwargs.model_id == EXPECTED_MODEL_ID
    assert dummy_experment_config.model_kwargs.model_type == EXPECTED_MODEL_TYPE
    assert dummy_experment_config.model_kwargs.batch_size == EXPECTED_BATCH_SIZE
    assert dummy_experment_config.model_kwargs.model_init_kwargs == EXPECTED_INIT_KWARGS


def test_PromptArguments(dummy_experment_config: ExperimentConfig):
    assert dummy_experment_config.prompt_kwargs.prompt_cls == EXPECTED_PROMPT_CLS


def test_model_dump(dummy_experment_config: ExperimentConfig):
    expected_kwargs = {
        "data_kwargs": {
            "data_path": EXPECTED_DATA_PATH,
            "schema_path": EXPECTED_SCHEMA_PATH,
            "argument_extraction": EXPECTED_ARG_EXTRACTION,
            "shuffle": EXPECTED_SHUFFLE,
        },
        "model_kwargs": {
            "model_id": EXPECTED_MODEL_ID,
            "model_type": EXPECTED_MODEL_TYPE,
            "batch_size": EXPECTED_BATCH_SIZE,
            "model_init_kwargs": EXPECTED_INIT_KWARGS,
        },
        "prompt_kwargs": {
            "prompt_cls": EXPECTED_PROMPT_CLS,
            "prompt_kwargs": EXPECTED_PROMPT_KWARGS,
        },
        "output_path": EXPECTED_OUTPUT,
    }
    assert dummy_experment_config.to_dict() == expected_kwargs
