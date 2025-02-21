# ARC-Event-Extraction

This project examines a range of state-of-the-art approaches for labeling events in text.

## Installation

As of the time of writing the project required python 3.12.

If Ollama models are to be used for prompting, then [Ollama](https://ollama.com/) must be installed.

### Using `pip`

```bash
git clone https://github.com/alan-turing-institute/ARC-Event-Extraction/
cd ARC-Event-Extraction
python -m venv .venv
source .venv/bin/activate
pip install -e arcee/.
pip install -e poc/.
pip install -e from_textee/.
pip install -e .
```

### Using `uv`

We are using `uv` workspaces, and also the uv.lock file to manage the dependencies.

```bash
git clone https://github.com/alan-turing-institute/ARC-Event-Extraction/
cd ARC-Event-Extraction
uv venv --python 3.12
uv sync
source .venv/bin/activate
```

If you are developing you will want to call `uv lock` to update the lock file.
This changes the python package versions to the latest available versions that are compatible.

### Install Pre-Commit Hooks

Once you have set up your virtual environment and installed everything, if doing development on the repository you will need to install pre-commit hooks.

```bash
pre-commit install
```


## Project Structure

This project has sub-packages that are used for different purposes.

### `arcee` - ARC Event Extraction

This is the main experiment package for the ARC Event Extraction project. [Read Me Here](arcee/README.md)

### `poc` - Proof of Concept

This is a proof of concept for the RAG prompting set up for the event extraction task. [Read Me Here](poc/README.md)

### `from_textee` - From TextEE

This package has a set of scripts taken and updated from the [TextEE project](https://github.com/ej0cl6/TextEE). [Read Me Here](from_textee/README.md)

## Scripts

The scripts folder contain scripts to run the following processes:

### `data_prep`
The scripts in this folder process the data.
To run `process_rams.py`, the following args could be used:
`-i=data/RAMS/data -o=data/RAMS/output/split1 -s=data/RAMS/splits/split1`

The script `create_dev_datasets.py` can be used to create a `train` and `dev` datasets that would specifically be used for prompt development, where the records come from an intersection of all five of the existing train splits (the number of splits can be altered, but there will be five if TextEE is followed). This can be run with the following args:
`-o=data/RAMS/output/ --num_splits=5 --seed=42`

### `objective_1`
The script in this folder is designed to try out different language models for event extraction tasks using in-context learning (ICL), i.e., without any fine tuning.
