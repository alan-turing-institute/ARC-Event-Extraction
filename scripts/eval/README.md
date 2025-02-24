# Evaluation

All evaluation in `arcee` is run using the [inference.py](/scripts/eval/inference.py) script. To run evaluation using this script, you will require a yaml config file, such as [example_job.yaml](/scripts/configs/example_job.yaml). To run inference using a given config file, run the following with the source code installed:

```bash
python scripts/eval/inference.py --config_path scripts/configs/example_job.yaml
```

## HPC

We use the Baskerville HPC to run our jobs, which makes use of a slurm. To use this, you will require a submission script, suchas [example_eval_submit.sh](/scripts/eval/example_eval_submit.sh). To run a slurm job, you can simply run:

```bash
sbatch example_eval_submit.sh
```

## Configs

Configs are loaded in and managed by the `arcee.config.experiment.ExperimentConfig` class. This contains as attributes config classes for the dataset, model, and prompt.

### Dataset

Dataset arguments are managed by a `arcee.config.data_arguments.DataArguments` class, which is stored in `ExperimentConfig.data_kwargs`.

The `data_kwargs` section of the config YAML file should contain the following:

- `data_path`: A path to an event extraction dataset stored in a .jsonlines format
- `schema_path`: A path to an event extraction schema

It can also optionall contain the following:

- `shuffle`: A bool denoting whether to shuffle examples in the dataloader. Defaults to `False`

### Model

Model arguments are managed by a `arcee.config.model_arguments.ModelArguments` class, which is stored in `ExperimentConfig.model_kwargs`.

The `model_kwargs` section of the config YAML file should contain the following:

- `model_id`: A string denoting the HuggingFace or Ollama model ID or path
- `model_type`: A string denoting whether to use a Huggingface (`"hf"`) or Ollama model (`"ollama"`)
- `batch_size`: An integer denoting the batch size to use during evaluation

It can optionally contain the following arguments:

- `model_init_kwargs`: Contains values specific to intialising either a HuggingFace or Ollama model. See relevant classes

### Prompt

Prompt arguments are managed by a `arcee.config.data_arguments.PromptArguments` class, which is stored in `ExperimentConfig.prompt_kwargs`.

The `prompt_kwargs` section of the config YAML file should contain the following:

- `prompt_cls`: A string which matches a prompt class in `arcee.prompts.prompts`
