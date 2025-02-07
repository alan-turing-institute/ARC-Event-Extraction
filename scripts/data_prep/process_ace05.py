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
    parser.add_argument(
        "--sent_map", required=True, default="sent_map.json from TextEE"
    )
    parser.add_argument(
        "--token_map", required=True, default="token_map.json from TextEE"
    )
    parser.add_argument(
        "--time_and_val", action="store_true", help="Extracts times and values"
    )
    parser.add_argument(
        "--full_span", action="store_true", help="Use full instead of head span"
    )
    args = parser.parse_args()

    data_prep.process_ace05_data(
        args.in_folder,
        args.out_folder,
        args.split_folder,
        args.sent_map,
        args.token_map,
        args.time_and_val,
        args.full_span,
    )


if __name__ == "__main__":
    main()
