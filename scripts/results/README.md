`analysis.py` takes in a YAML config and computes results tables. This assumes that you have previously run event extraction jobs, produce their outputs, and computed model performances using `scripts/stats/generate_stats.py <OUTPUT_DIR_NAME>`.

The YAML file should contain the following fields:

- `name`: A name defining the output file. As written, should be one of `icl` or `ft`: to extend this, you should modify `APPROACH_LABEL_MAP` on line 55
- `run_groups`: A list of run groups, defining criteria for different runs to include in your output. All run groups will be combined for the purpose of the final output, so where different tables are desired, different YAML files should be used.

A single `run_group` should contain the following:

- `folder`: A list of output directory names containing runs you wish to compute results over
- `model`: A list of model names which you wish to compute results over. Note that you may need to extend this list in `MODEL_SIZE_MAP` on line 31 if introducing models we did not use
- `constraint`: A list of generation constraints, as captured in the names of wandb runs. Should be one of `f` for free, `j` for JSON conraints, and `s` for schema based
- `ed` and `eae`: should contain a variable named `prompt`, which in turn contains a list of prompt types
- `num_example`: A list of the number of examples used, as capture in a prompt name

For an example, see `scripts/configs/results/icl_results.yaml`
