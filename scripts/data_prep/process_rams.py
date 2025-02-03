"""Processing RAMS dataset adapted from TextEE

This script processes the RAMS dataset to the format used in TextEE making use of the
from_textee package. The script requires a few arguments to run. An example of these
arguments is shown below:

`-i=data/RAMS/data -o=data/RAMS/output/split1 -s=data/RAMS/splits/split1`
"""

from __future__ import annotations

from argparse import ArgumentParser

from from_textee import data_prep


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--in_folder", required=True, help="Path to the input folder"
    )
    parser.add_argument(
        "-o", "--out_folder", required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "-s", "--split_folder", required=True, help="Path to split folder"
    )
    args = parser.parse_args()

    data_prep.process_rams_data(args.in_folder, args.out_folder, args.split_folder)


if __name__ == "__main__":
    main()
