# From TextEE

This package contains scripts from the package TextEE, which have been adapted for our use case.
<https://github.com/ej0cl6/TextEE>
Various scripts have been copied, rather than cloning the whole repo as not all the scripts are required and some structural changes have been made to the code.

A summary of the changes that have been made to the TextEE scripts:

- Unused requirements were removed from the `.toml` file
- Scripts were updated to adhere to linters, e.g. Ruff
- The arguments to the scripts were slightly updated.

## Data Preparation

The data preparation scripts are taken from the TextEE repo and a module has been created to replicate the functionality. These scripts are used to prepare the data from different sources into a consistent format for the model to use. Not every dataset that features in TextEE will be used in this project, so only the functionallity for RAMS, WikiEvents, and ACE05 will be implemented.

The data will be saved in JSON lines format. The format that the JSONs need to be in is as follows:

- "doc_id" (`str`): Unique identifier for the document.
- "wnd_id" (`str`): Window identifier for the document.
- "text" (`str`): Full text of the document.
- "tokens" (`list[str]`): List of tokens in the document.
- "event_mentions" (`list[dict]`): List of event mentions, each with the following keys:
    * "id" (`str`): Unique identifier for the event.
    * "event_type" (`str`): Type of the event.
    * "trigger" (`dict`): Trigger information with keys:
        - "text" (`str`): Text of the trigger.
        - "start" (`int`): Start index of the trigger.
        - "end" (`int`): End index of the trigger.
    * "arguments" (`list[dict]`): List of arguments, each with keys
        - "entity_id" (`str`): Unique identifier for the entity.
        - "role" (`str`): Role of the entity.
        - "text" (`str`): Text of the entity.
        - "start" (`int`): Start index of the entity.
        - "end" (`int`): End index of the entity.
- "entity_mentions" (`list[dict]`): List of entity mentions, each with the following keys:
    * "id" (`str`): Unique identifier for the entity.
    * "text" (`str`): Text of the entity.
    * "entity_type" (`str`): Type of the entity.
    * "start" (`int`): Start index of the entity.
    * "end" (`int`): End index of the entity.
- "lang" (`str`): Language of the document. Always 'en'.

The "start" and "end" indices are inclusive and exclusive respectively and refer to the tokens in the "tokens" list.

The TextEE description of the data also can contain a "relation_mentions" field. This doesn't seem to be needed downstream, so it will be ignored.
