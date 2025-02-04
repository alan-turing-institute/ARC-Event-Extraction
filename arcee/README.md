# ARCEE

The ARC Event Extractor (arcee) is a experimental set about extracting events.

## Prompts
The templates for the prompts that have been used for the experimental work are stored in the `prompts/prompts.py`. There is a base class `EEPrompt` which all other prompt classes inherit from. The base class has two methods, which could be implemented differently by each class that inherits:
- `get_system_content` - this returns a message for the system content for a prompt message
- `get_prompt` - this takes some text as input, from when events (or their arguments) are to be extracted. The method returns a prompt and a response format (e.g., "json") that can be handed to a language model.

Currently, only one prompt class has been implemented:
- `EAEPromptNoExample_01` - this returns a prompt for EAE without giving any examples and does not constrain the response of the model to JSON

Further prompt classes will follow, and we expect that the logic for retrieving examples for the prompt to sit within these classes.

## EE Models
The only class that has been created to date is:
- `EEOllamaModel` - a wrapper class for Ollama Models

If Ollama models are to be used, then [Ollama](https://ollama.com/) must be installed.
