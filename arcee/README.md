# ARCEE

The ARC Event Extractor (arcee) is a experimental set about extracting events.

## Prompts
The templates for the prompts that have been used for the experimental work are stored in the `prompts/prompts.py`. This module provides classes for constructing prompts for event extraction tasks. The prompts are designed to be used with a language model to extract event triggers and arguments from input text.

There is a base class `EEPrompt` which all other prompt classes inherit from. The base class has two methods, which could be implemented differently by each class that inherits:
- `_get_system_content(self) -> str`: Returns the system content message to be used. This method can be overwritten by a child class.
- `_construct_prompt(self, data: dict[str, str | int | list], schema: dict) -> tuple[str, str | dict, dict]`: Abstract method for constructing the prompt.
- `get_prompt(self, data: dict[str, str | int | list], schema: dict, **kwargs: str | int) -> tuple[str, str | dict, dict]`: Returns a prompt. This method can be overwritten by child classes.

### Prompt classes for event trigger detection

#### `EDPrompter1`

A class for constructing prompts for event trigger detection. This class does not use any examples in the prompt and requests JSON output with a description of the schema.

#### `EDPrompter2`

A class for constructing prompts for event trigger detection. This class does not use any examples in the prompt and requests JSON output with a defined JSON schema.

#### `EDPrompter3`
A class for constructing prompts for event trigger detection. This class uses random examples in the prompt (the number determined by an input argument) and requests JSON output with a defined JSON schema.

#### `EDPrompter4`
A class for constructing prompts for event trigger detection. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity) and requests JSON output with a defined JSON schema.

### Prompt classes for event argument extraction

#### `EAEPrompter1`

A class for constructing prompts for event argument extraction. This class only takes a single event trigger, predicts multiple argument roles, does not use examples in the prompt, and requests JSON output.

#### `EAEPrompter2`

A class for constructing prompts for event argument extraction. This class only takes a single event trigger, predicts multiple argument roles, does not use examples in the prompt, and requests JSON output with a specific schema.


## EE Models
The only class that has been created to date is:
- `EEOllamaModel` - a wrapper class for Ollama Models

If Ollama models are to be used, then [Ollama](https://ollama.com/) must be installed.
