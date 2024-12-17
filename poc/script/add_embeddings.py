from __future__ import annotations

import os

import pandas as pd
from sentence_transformers import SentenceTransformer

ORIGINAL_DATA_PATH = "data/gpt_data.json"
SAVE_DATA_PATH = "data/gpt_data_with_embeddings.json"
TRANSFORMER_MODEL = "all-mpnet-base-v2"


def main() -> None:
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    gpt_example_df = pd.read_json(
        os.path.join(main_dir, ORIGINAL_DATA_PATH),
    )

    # Run the embedding model
    st_model = SentenceTransformer(TRANSFORMER_MODEL)
    gpt_example_df["Embedding"] = gpt_example_df["Sentence"].map(st_model.encode)

    # Save the data
    gpt_example_df = gpt_example_df.reset_index().rename(columns={"index": "Index"})
    gpt_example_df.to_json(
        os.path.join(main_dir, SAVE_DATA_PATH), orient="records", lines=True
    )


if __name__ == "__main__":
    main()
