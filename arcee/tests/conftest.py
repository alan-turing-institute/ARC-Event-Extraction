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
