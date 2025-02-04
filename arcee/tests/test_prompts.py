from __future__ import annotations

import pytest

from arcee.prompts.prompts import EAEPromptNoExample_01


def test_EAEPromptNoExample_01(
    test_tokens: str, test_event_mentions: list, test_schema: dict
):
    expected_prompt = (
        "An event type of hire has been detected in the input text. "
        "The trigger word is identified with a <trigger> tag.\n "
        "Input text:\n```The university <trigger>hired</trigger> Dr . "
        "Smith to lead the biology department .```\n What are the words "
        "in the input text that represent the argument role ```object```?"
    )
    eae_prompt = EAEPromptNoExample_01()

    prompt, response_type = eae_prompt.get_prompt(
        test_tokens, test_event_mentions, test_schema, []
    )

    assert prompt == expected_prompt

    assert response_type == ""

    # Test that even if examples are provided, the prompt won't change
    with pytest.warns(UserWarning):
        prompt, response_type = eae_prompt.get_prompt(
            test_tokens, test_event_mentions, test_schema, ["one", "two", "three"]
        )
    assert prompt == expected_prompt
    assert response_type == ""
