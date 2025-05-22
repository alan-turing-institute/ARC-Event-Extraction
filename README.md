# ARC-Event-Extraction

This project examines a range of state-of-the-art approaches for labelling events in text.

## Installation

As of the time of writing, this project requires Python 3.12.

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

If you are developing, you will want to call `uv lock` to update the lock file.
This changes the python package versions to the latest available versions that are compatible.

### Install Pre-Commit Hooks

Once you have set up your virtual environment and installed everything, if doing development on the repository, you will need to install pre-commit hooks.

```bash
pre-commit install
```

### Optional CUDA Dependencies

If the quantised, fine-tuned models are run, this requires the `bitsandbytes` package, which can only work with CUDA.
If you want to install this extra dependency manually with pip, or with `uv`, you can run:

```bash
uv sync --all-extras
```

## Project Structure

*This project comprises the following sub-packages*, each serving distinct purposes.
These can be installed independly if only some of the functionality is needed.

### `arcee` - ARC Event Extraction

This is the main experiment package for the ARC Event Extraction project. [Read Me Here](arcee/README.md)

### `poc` - Proof of Concept

This is a proof of concept for the RAG prompting set-up for the event extraction task. [Read Me Here](poc/README.md)

### `from_textee` - From TextEE

This package has a set of scripts taken and updated from the [TextEE project](https://github.com/ej0cl6/TextEE). [Read Me Here](from_textee/README.md)

*The main interface* for this project is contained in the scripts folder.

### `scripts`

The scripts folder contains scripts which are broken down into four groups.
The `from_textee` package is needed for some of the intishal processes in `data_prep`, and the `arcee` package is needed for everything else.

#### `data_prep`

The scripts in this folder process the data.
To run `process_rams.py`, the following args could be used:
`-i=data/RAMS/data -o=data/RAMS/output/split1 -s=data/RAMS/splits/split1`

The script `create_dev_datasets.py` can be used to create a `train` and `dev` datasets that would specifically be used for prompt development, where the records come from an intersection of all five of the existing train splits (the number of splits can be altered, but there will be five if TextEE is followed). This can be run with the following args:
`-o=data/RAMS/output/ --num_splits=5 --seed=42`

#### `eval`

The scripts in this folder run the models on the data.
The `inference.py` script is run by providing a config from the `configs` folder.
It also contains an example Slurm script.
More information is provided in [the evaluation readme](scripts/eval/README.md).

#### `fine_tune`

The [script `run_fine_tune.py`](scripts/fine_tune/run_fine_tune.py) is used to fine-tune the model for a particular task.
The script takes one argument, which is the path to a config in the `fine_tune/configs` folder.
This can be run without using quantisation, but if it is required, the optional `cuda` dependency group needs to be installed.

The adapters for the models that are fine-tuned are saved to the `models` folder.

#### `stats`

 There are four scripts in the `scripts/stats` folder.

 * `generate_stats.py` - This generates the main statistics for the report and runs through the collection of results in the output folder and saves it to Weights and Biases.
 * `f1_with_variance.py` - Generates the results to be shown in the figures, which can also save a collection of metadata and uses bootstrapping to calculate a confidence interval to display.
 * `compare_prediction.py` -  This is used to compare the outputs of each of the models with each other by changing what is considered the ground truth.
 * `view_examples.py` - This displays outputs of the model in a readable way and can be filtered on different criteria.

*The following folders* contain the input and output files required for the project.

### `data`

This folder stores the raw and pre-processed data for use by the event extraction models.

### `models`

This folder is used to contain the language models used in the report and the adapters produced by the fine-tuning scripts.

### `outputs`

This folder stores the outputs from the models.
This folder's structure follows the form found in the `scripts/configs` folder.

### `stats`

This folder stores the statistics about model performance.
It also stores the configs that are used by the files in `scripts/stats` to decide what is generated.

### `report-figures`

This folder contains the scripts and data used to produce the figures for the report.
