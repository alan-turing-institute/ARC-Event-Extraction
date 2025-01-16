import os

import pandas as pd
from numpy.random import RandomState

DATA_PATH = "data/gpt_data.json"  # Path to the data file
SAVR_PATH = "data/gpt_"  # Save path for the train, test, and eval DataFrames
RANDOM_SEED = 42  # Random seed for reproducibility


def check_inclusion(row):
    """Check if the EventTrigger, Subject, and Object are in the sentence.
    Will print an message for each stennce where this isn't followed"""
    sentence = row["Sentence"]
    for col in ["EventTrigger", "Subject", "Object"]:
        if row[col] and row[col] not in sentence:
            print(
                f"Row {row.name} - {col}:'{row[col]}' - " f"not in :: {row['Sentence']}"
            )


def main():
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    random_state = RandomState(RANDOM_SEED)

    data_df = pd.read_json(os.path.join(main_dir, DATA_PATH))

    # Apply the checking function to each sentence
    data_df.apply(check_inclusion, axis=1)

    train_df_list = []
    test_df_list = []
    eval_df_list = []

    for event_type in data_df["EventType"].unique():
        event_data = data_df[data_df["EventType"] == event_type]
        shuffled_data = event_data.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        # This assumes there are 36 samples for each event type
        train_data = shuffled_data.iloc[:24]  # 24 samples for training
        test_data = shuffled_data.iloc[24:34]  # 10 samples for testing
        eval_data = shuffled_data.iloc[34:]  # 2 samples for evaluation
        train_df_list.append(train_data)
        test_df_list.append(test_data)
        eval_df_list.append(eval_data)

    train_df = pd.concat(train_df_list).reset_index(drop=True)
    test_df = pd.concat(test_df_list).reset_index(drop=True)
    eval_df = pd.concat(eval_df_list).reset_index(drop=True)

    train_df.to_csv(
        os.path.join(main_dir, SAVR_PATH + "train.tsv"), index=False, sep="\t"
    )
    test_df.to_csv(
        os.path.join(main_dir, SAVR_PATH + "test.tsv"), index=False, sep="\t"
    )
    eval_df.to_csv(
        os.path.join(main_dir, SAVR_PATH + "eval.tsv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    main()
