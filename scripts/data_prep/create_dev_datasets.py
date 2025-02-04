"""
This script creates a train and dev dataset that will be used for development only -
probably these will only be used during prompt development. They will be created
from the intersection of the train files in all the splits for a given datasets
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser

import numpy as np
from arcee.utils.data import load_data


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-o", "--out_folder", required=True, help="Path to the output folder"
    )
    parser.add_argument("-n", "--num_splits", default=5, help="Number of splits to use")
    parser.add_argument(
        "--seed", default=0, help="Random seed to determine train / dev split"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    num_splits = args.num_splits
    output_path = args.out_folder

    data = []
    # Assume first split is numbered 1
    for i in range(1, num_splits + 1):
        fileloc = os.path.join(output_path, "split" + str(i), "train.jsonlines")
        data.append(load_data(fileloc))

    data_intersection = data[0]
    for i in range(1, num_splits):
        data_intersection = [x for x in data_intersection if x in data[i]]

    np.random.shuffle(data_intersection)

    split_id_float = 9 * len(data_intersection) / 10
    split_id = round(split_id_float)

    new_split_foler = os.path.join(output_path, "split_dev")
    if not os.path.exists(new_split_foler):
        os.makedirs(new_split_foler)

    with open(os.path.join(new_split_foler, "train.jsonlines"), "w") as fp:
        for data in data_intersection[:split_id]:
            fp.write(json.dumps(data) + "\n")

    with open(os.path.join(new_split_foler, "dev.jsonlines"), "w") as fp:
        for data in data_intersection[split_id:]:
            fp.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
