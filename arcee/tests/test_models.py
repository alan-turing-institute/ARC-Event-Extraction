"""
Placeholder for model tests.
"""

from __future__ import annotations

from arcee.ee_models.ee_models import EEMockOllamaModel


def test_OllamaPredict(test_batch_prompts: list[dict[str, str]]):
    eemodel = EEMockOllamaModel()
    output = eemodel.predict(test_batch_prompts)

    assert output == ['{"object": "mock object"}', '{"object": "mock object"}']
