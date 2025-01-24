# ARC-Event-Extraction

The Event Extraction project.


## Installation

As of the time of writing the project required python >3.10 <3.12

Using `pip`

```bash
git clone https://github.com/alan-turing-institute/ARC-Event-Extraction/
cd ARC-Event-Extraction
pip install -e poc/.
pip install -e from_textee/.
pip install -e .
```

Using `uv`
We are using `uv` workspaces, and also the uv.lock file to manage the dependencies.

```bash
git clone https://github.com/alan-turing-institute/ARC-Event-Extraction/
cd ARC-Event-Extraction
uv venv
uv sync
```

If you are developing you will want to call `uv lock` to update the lock file.
This changes the python package versions to the latest available versions that are compatible with the current versions.
