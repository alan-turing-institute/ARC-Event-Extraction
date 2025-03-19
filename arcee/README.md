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

#### `EAEPromptWithExampleSchema`

A class that adds an example of the required EAE JSON schema into the prompt. Inherits from `EAPrompt`.

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

#### `EDPrompter5`
A class for constructing prompts for event trigger detection. This class uses both random and similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). The random examples are displayed first in the prompt and there are the same number of random and similar examples in the prompt.

#### `EDPrompter6`
A class for constructing prompts for event trigger detection. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). The named entities in the input text and example text are masked before comparing for similarity.

#### `EDPrompter7`
A class for constructing prompts for event trigger detection. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). The most similar examples are used to filter the possible event types shown in the prompt, so that the prompt only contains those event types common to the similar examples - if there are fewer than two event types then additional event types are added to the prompt.

#### `EDPrompter8`
A class for constructing prompts for event trigger detection. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). As with `EDPrompter7`, this prompt also filters by event type but the similar examples are selected by masking the named entities.

### Prompt classes for event argument extraction

#### `EAEPrompter1`

A class for constructing prompts for event argument extraction. This class only takes a single event trigger, predicts multiple argument roles, does not use examples in the prompt.

#### `EAEPrompter2`

A class for constructing prompts for event argument extraction. This class does not use any examples in the prompt and describes the required schema and provides an explicit example of the format.

#### `EAEPrompter3`
A class for constructing prompts for event argument extraction. This class uses random examples in the prompt (the number determined by an input argument).

#### `EAEPrompter4`
A class for constructing prompts for event argument extraction. This class uses random examples in the prompt (the number determined by an input argument). The random examples are chosen from the examples with the same event type as the input record; if there are not enough examples with the same event type to choose from, then examples will be selected at random (as per `EAEPrompter3`).

#### `EAEPrompter5`
A class for constructing prompts for event argument extraction. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity).

#### `EAEPrompter6`
A class for constructing prompts for event argument extraction. This class uses random examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). The similar examples are chosen from the examples with the same event type as the input record; if there are not enough examples with the same event type to choose from, then examples will be selected as the most similar, regardless of event type (as per `EAEPrompter5`).

#### `EAEPrompter7`
A class for constructing prompts for event argument extraction. This class uses both random and similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). The random examples are displayed first in the prompt and there are the same number of random and similar examples in the prompt.

#### `EAEPrompter8`
A class for constructing prompts for event argument extraction. This class uses similar examples in the prompt (the number determined by an input argument and the similarity determined by a sentence transformer using cosine similarity). The named entities in the input text and example text are masked before comparing for similarity.

## EE Models
There are different frameworks that can be used to run LLMs: Ollama, HuggingFace and Azure OpenAI. Additionally, there is the option to use the guidance library (TODO: link?) with models that are implemented with HuggingFace.

The following classes have been implemented:
#### `EEOllamaModel`
A wrapper class for Ollama Models
If Ollama models are to be used, then [Ollama](https://ollama.com/) must be installed.

#### `EEHFModel`
A wrapper class for using Huggingface transformers.

#### `EEOpenAIModel`
A wrapper class for the Azure OpenAI models. To use these models, the following environment variables must be set:
```
export AZURE_OPENAI_API_KEY="REPLACE_WITH_YOUR_KEY_VALUE_HERE"
export AZURE_OPENAI_ENDPOINT="REPLACE_WITH_YOUR_ENDPOINT_HERE"
```
