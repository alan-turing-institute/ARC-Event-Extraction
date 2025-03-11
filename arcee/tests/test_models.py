from __future__ import annotations

import pytest

from arcee.ee_models.ee_models import EEMockOllamaModel, EEOllamaModel


def test_ollama_output_format():
    """
    Tests that if an output format other than '', 'json' or 'json_schema' is provided,
    an error is raised.
    """
    output_format = "invalid_format"
    err_msg = "output_format must be one of '', 'json', or 'json_schema'."
    with pytest.raises(ValueError, match=err_msg):
        EEOllamaModel(model_id="test", output_format=output_format)
        # assert str(e) == "output_format must be one of '', 'json', or 'json_schema'."


@pytest.mark.parametrize("output_format", [(""), ("json"), ("json_schema")])
def test_ollama_predict(test_batch_prompts, output_format: str):
    """
    Tests that the Ollama model can predict on a batch of prompts.
    """
    batch = []
    for idx, prompt in enumerate(test_batch_prompts):
        batch.append((prompt, {"mock_json_schema"}, {"id": str(idx)}))

    response_format = (
        {"mock_json_schema"} if output_format == "json_schema" else output_format
    )

    model = EEMockOllamaModel(output_format=output_format)
    predictions = model.predict(batch)
    assert len(predictions) == len(batch)
    assert predictions[0] == {
        "id": "0",
        "output": {
            "response": "mock object with output format " + str(response_format)
        },
    }
