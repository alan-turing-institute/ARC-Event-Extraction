from __future__ import annotations

import argparse
import json
import os

from arcee.dataset.schemas import data_to_argument_schema, data_to_ed_schema


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate gold data for a specified dataset."
    )
    parser.add_argument(
        "dataset_name", type=str, help="The name of the dataset (e.g., 'ACE05')"
    )
    args = parser.parse_args()

    # Define the main directory path (two levels above the current file)
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    dataset_dir = os.path.join(main_dir, f"data/{args.dataset_name}")
    # Define the paths to the train, dev, and test jsonlines files based on the dataset
    # name
    data_paths = [
        os.path.join(dataset_dir, "output/split1/train.jsonlines"),
        os.path.join(dataset_dir, "output/split1/dev.jsonlines"),
        os.path.join(dataset_dir, "output/split1/test.jsonlines"),
    ]

    # Load the data from the jsonlines files
    combined_data = []
    for data_path in data_paths:
        with open(data_path) as data_file:
            combined_data.extend([json.loads(line) for line in data_file])

    # Pattern file
    schema_path = os.path.join(dataset_dir, "schema/pattern.json")

    # Pass the combined data to the data_to_argument_schema function
    argument_schema_data = data_to_argument_schema(combined_data, schema_path)

    # Pass the combined data to the data_to_ed_schema function
    ed_schema_data = data_to_ed_schema(combined_data)

    # Define the output directory path
    output_dir = os.path.join(dataset_dir, "gold")
    os.makedirs(output_dir, exist_ok=True)
    # Save the argument schema data to a json lines file
    with open(os.path.join(output_dir, "gold_label_args.jsonl"), "w") as output_file:
        for line in argument_schema_data:
            output_file.write(json.dumps(line) + "\n")
    # Save the ED schema data to a json lines file
    with open(
        os.path.join(output_dir, "gold_label_triggers.jsonl"), "w"
    ) as output_file:
        for line in ed_schema_data:
            output_file.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
