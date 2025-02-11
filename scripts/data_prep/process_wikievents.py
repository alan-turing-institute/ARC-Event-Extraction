"""Processing WikiEvents dataset adapted from TextEE

This script processes the WikiEvents dataset to the format used in TextEE making use
of the from_textee package. The script requires a few arguments to run. An example of
these aregumens is shown below:

`
-i=data/WikiEvents/data
-o=data/WikiEvents/output/split1
-s=data/WikiEvents/splits/split1
--seg_map=data/WikiEvents/segment_map.json
`

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
    parser.add_argument("--seg_map", required=True, help="Path to segment map")
    args = parser.parse_args()

    data_prep.process_wikievents_data(
        args.in_folder, args.out_folder, args.split_folder, args.seg_map
    )


if __name__ == "__main__":
    main()
