# from __future__ import annotations

# # import pytest
# from arcee.prompts.prompts import DummyEAEPrompter


# def test_DummyEAEPrompter(
#     test_tokens: str, test_event_mentions: list, test_schema: dict
# ):
#     expected_prompt = (
#         "An event type of hire has been detected in the input text. "
#         "The trigger word is identified with a <trigger> tag.\n "
#         "Input text:\n```The university <trigger>hired</trigger> Dr . "
#         "Smith to lead the biology department .```\n What are the words "
#         "in the input text that represent the argument role ```object```? "
#         "Provide your response in JSON format with the "
#         "key object and the value as the words in the input text "
#         "that represent the argument."
#     )
#     eae_prompt = DummyEAEPrompter()

#     prompt = eae_prompt.get_prompt(test_tokens, test_event_mentions, test_schema)

#     assert prompt == expected_prompt
