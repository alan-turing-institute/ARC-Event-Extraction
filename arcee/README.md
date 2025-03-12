# ARCEE

The ARC Event Extractor (arcee) is a experimental set about extracting events.

## Prompts
The templates for the prompts that have been used for the experimental work are stored in the `prompts/prompts.py`. This module provides classes for constructing prompts for event extraction tasks. The prompts are designed to be used with a language model to extract event triggers and arguments from input text.

The abstract classes contain all the functionality, and the concrete classes use multiple inheritence to determine their implementation.

There is a base class `EEPrompt` which all other prompt classes inherit from. The class contains the only method that is intended to be called externally:
- `get_prompt(self, data: dict[str, str | int | list], schema: dict, **kwargs: str | int) -> tuple[str, dict, dict]`: Returns a prompt.
The `get_prompt` method takes a data record and returns a prompt, a dictionary representing a JSON schema that is the expected output format from the model and a dictionary containing a unique identifier for the data record.

The other abstract classes are:
#### `EDPrompt`

A class containing common functionality for all event detectiction prompts. Inherits from `EEPrompt`.

#### `EDPromptWithExampleSchema`

A class that adds an example of the required ED JSON schema into the prompt. Inherits from `EDPrompt`.

#### `EAEPrompt`

A class containing common functionality for all event argument extraction prompts. Inherits from `EEPrompt`.

#### `EEPromptWithExamples`

A class containing common functionality for all prompts that include examples - common to both ED and EAE implementations. Inherits from `EEPrompt`.

#### `EEPromptRandomExamples`

A class containing common functionality for all prompts that includ RANDOM examples - common to both ED and EAE implementations. Inherits from `EEPromptWithExamples`.

#### `EEPromptSimilarExamples`

A class containing common functionality for all prompts that includ SIMILAR examples, determined using a sentence transformer model - common to both ED and EAE implementations. Inherits from `EEPromptWithExamples`.

#### `EDPromptWithExamples`

A class containing common functionality for all ED prompts that include examples. Inherits from `EEPromptWithExamples`.

#### `EAEPromptWithExamples`

A class containing common functionality for all EAE prompts that include examples. Inherits from `EEPromptWithExamples`.

### Prompt classes for event trigger detection

#### `EDPrompter1`

A class for constructing prompts for event trigger detection. This class does not use any examples in the prompt and describes the required schema without providing an explicit example of the format.

#### `EDPrompter2`

A class for constructing prompts for event trigger detection. This class does not use any examples in the prompt and describes the required schema and provides an explicit example of the format.

#### `EDPrompter3`
A class for constructing prompts for event trigger detection. This class uses random examples in the prompt (the number determined by an input argument).

#### `EDPrompter4`
A class for constructing prompts for event trigger detection. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity).

### Prompt classes for event argument extraction

#### `EAEPrompter1`

A class for constructing prompts for event argument extraction. This class only takes a single event trigger, predicts multiple argument roles, does not use examples in the prompt.

#### `EAEPrompter2`

This class is not implemented.

#### `EAEPrompter3`
A class for constructing prompts for event argument extraction. This class uses random examples in the prompt (the number determined by an input argument) and requests JSON output with a specific schema.

#### `EAEPrompter4`
A class for constructing prompts for event argument extraction. This class uses random examples in the prompt (the number determined by an input argument) and requests JSON output with a specific schema. The random examples are chosen from the examples with the same event type as the input record; if there are not enough examples with the same event type to choose from, then examples will be selected at random (as per `EAEPrompter3`).

#### `EAEPrompter5`
A class for constructing prompts for event argument extraction. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity) and requests JSON output with a defined JSON schema.

#### `EAEPrompter6`
A class for constructing prompts for event argument extraction. This class uses random examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity) and requests JSON output with a defined JSON schema. The similar examples are chosen from the examples with the same event type as the input record; if there are not enough examples with the same event type to choose from, then examples will be selected as the most similar, regardless of event type (as per `EAEPrompter5`).

## EE Models
The only class that has been created to date is:
- `EEOllamaModel` - a wrapper class for Ollama Models

If Ollama models are to be used, then [Ollama](https://ollama.com/) must be installed.
