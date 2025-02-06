from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def test_tokens() -> list[str]:
    return [
        "The",
        "university",
        "hired",
        "Dr",
        ".",
        "Smith",
        "to",
        "lead",
        "the",
        "biology",
        "department",
        ".",
    ]


@pytest.fixture(scope="module")
def test_event_mentions() -> list:
    return [{"event_type": "hire", "trigger": {"text": "hired", "start": 2, "end": 3}}]


@pytest.fixture(scope="module")
def test_schema() -> dict:
    return {"hire": {"roles": ["object", "subject"]}}


@pytest.fixture(scope="module")
def test_batch_prompts() -> list[dict[str, str]]:
    prompt1 = (
        "An event type of hire has been detected in the input text. "
        "The trigger word is identified with a <trigger> tag.\n "
        "Input text:\n```The university <trigger>hired</trigger> Dr . "
        "Smith to lead the biology department .```\n What are the words "
        "in the input text that represent the argument role ```object```? "
        "Provide your response in JSON format with the "
        "key object and the value as the words in the input text "
        "that represent the argument."
    )
    prompt2 = (
        "An event type of hire has been detected in the input text. "
        "The trigger word is identified with a <trigger> tag.\n "
        "Input text:\n```The university <trigger>hired</trigger> Dr . "
        "Jones to lead the physics department .```\n What are the words "
        "in the input text that represent the argument role ```object```? "
        "Provide your response in JSON format with the "
        "key object and the value as the words in the input text "
        "that represent the argument."
    )
    return [{"role": "user", "content": prompt1}, {"role": "user", "content": prompt2}]
