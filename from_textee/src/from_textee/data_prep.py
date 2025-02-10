from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import PosixPath

import numpy as np
from tqdm import tqdm

__all__ = [
    "read_docs",
    "save_data",
    "get_split",
    "print_statistics",
    "convert_rams_format",
]


def read_docs(path: str | PosixPath) -> list[dict]:
    """
    Reads and parses JSON lines files from the specified directory.
    This function reads three files: "train.jsonlines", "dev.jsonlines", and
    "test.jsonlines" from the given directory path. It parses each line in these files
    as a JSON object and aggregates all objects into a single list.
    Args:
        path (str): The directory path where the JSON lines files are located.
    Returns:
        list: A list of all JSON objects parsed from the three files.
    """

    all_objs = []
    with open(os.path.join(path, "train.jsonlines")) as fp:
        lines = fp.readlines()
    objs = [json.loads(line) for line in lines]
    all_objs.extend(objs)

    with open(os.path.join(path, "dev.jsonlines")) as fp:
        lines = fp.readlines()
    objs = [json.loads(line) for line in lines]
    all_objs.extend(objs)

    with open(os.path.join(path, "test.jsonlines")) as fp:
        lines = fp.readlines()
    objs = [json.loads(line) for line in lines]
    all_objs.extend(objs)

    return all_objs


def save_data(
    out_path: str | PosixPath,
    train_data: list[dict],
    dev_data: list[dict],
    test_data: list[dict],
) -> None:
    """
    Save training, development, and test data to JSON lines files.

    Args:
        out_path (str): The output directory path where the JSON files will be saved.
        train_data (list[dict]): A list of dictionaries containing the training data.
        dev_data (list[dict]): A list of dictionaries containing the development data.
        test_data (list[dict]): A list of dictionaries containing the test data.
    """
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "train.jsonlines"), "w") as fp:
        for data in train_data:
            fp.write(json.dumps(data) + "\n")

    with open(os.path.join(out_path, "dev.jsonlines"), "w") as fp:
        for data in dev_data:
            fp.write(json.dumps(data) + "\n")

    with open(os.path.join(out_path, "test.jsonlines"), "w") as fp:
        for data in test_data:
            fp.write(json.dumps(data) + "\n")


def get_split(
    objs: list[dict], split_folder: str | PosixPath, key_name: str = "doc_id"
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Splits a list of objects into training, development, and test sets based on
    document IDs.

    Args:
        objs (list[dict]): A list of dictionaries, each containing a "doc_key" that
            identifies the document.
        split_folder (str): The folder path where the split files
            ("train.txt", "dev.txt", "test.txt") are located.
        key_name (str, optional): The key name to use for identifying the document.
            Defaults to "doc_id".

    Returns:
        tuple[list[dict]]: A tuple containing three lists of dictionaries:
            - The first list contains the training objects.
            - The second list contains the development objects.
            - The third list contains the test objects.
    """

    with open(os.path.join(split_folder, "train.txt")) as fp:
        lines = fp.readlines()
        train_doc_ids = {*[line.strip() for line in lines]}

    with open(os.path.join(split_folder, "dev.txt")) as fp:
        lines = fp.readlines()
        dev_doc_ids = {*[line.strip() for line in lines]}

    with open(os.path.join(split_folder, "test.txt")) as fp:
        lines = fp.readlines()
        test_doc_ids = {*[line.strip() for line in lines]}

    train_objs = []
    dev_objs = []
    test_objs = []

    for obj in objs:
        if obj[key_name] in train_doc_ids:
            train_objs.append(obj)
        if obj[key_name] in dev_doc_ids:
            dev_objs.append(obj)
        if obj[key_name] in test_doc_ids:
            test_objs.append(obj)

    train_objs.sort(key=lambda x: x[key_name])
    dev_objs.sort(key=lambda x: x[key_name])
    test_objs.sort(key=lambda x: x[key_name])

    return train_objs, dev_objs, test_objs


def print_statistics(data: list[dict]) -> None:
    """
    This takes a list of event objects in the TextEE format and prints statistics
    about the data.

    Args:
        data (list[dict]): A list of dictionaries, holding event objects in the TextEE
            format.
    """
    event_type_count: dict[str, int] = defaultdict(int)
    role_type_count: dict[str, int] = defaultdict(int)
    doc_ids = set()
    max_len = 0
    for dt in data:
        max_len = max(max_len, len(dt["tokens"]))
        doc_ids.add(dt["doc_id"])
        for event in dt["event_mentions"]:
            event_type_count[event["event_type"]] += 1
            for argument in event["arguments"]:
                role_type_count[argument["role"]] += 1

    print(f"# of Instances: {len(data)}")
    print(f"# of Docs: {len(doc_ids)}")
    print(f"Max Length: {max_len}")
    print(f"# of Event Types: {len(event_type_count)}")
    print(f"# of Events: {sum(event_type_count.values())}")
    print(f"# of Role Types: {len(role_type_count)}")
    print(f"# of Arguments: {sum(role_type_count.values())}")
    print()


def convert_rams_format(objs: list[dict]) -> list[dict]:
    """
    Convert a list of event objects from the RAMS format to the format used in TextEE.

    Args:
        objs (list[dict]): A list of dictionaries, each representing a document with
            the following keys:
            - "doc_key" (str): Unique identifier for the document.
            - "sentences" (list[list[str]]): List of sentences, where each sentence is
                a list of tokens.
            - "ent_spans" (list[tuple[int, int]]): List of entity spans.
            - "evt_triggers" (list[tuple[int, int, list[list[str]]]]): List of event
                triggers, where each trigger is a tuple of start index, end index,
                and event type.
            - "gold_evt_links" (list[tuple[tuple[int, int], tuple[int, int], str]]):
                List of event links, where each link is a tuple of trigger span,
                argument span, and role.

    Returns:
        list[dict]: A list of dictionaries, each representing a document in the
            converted format with the following keys:
            - "doc_id" (str): Unique identifier for the document.
            - "wnd_id" (str): Window identifier for the document.
            - "text" (str): Full text of the document.
            - "tokens" (list[str]): List of tokens in the document.
            - "event_mentions" (list[dict]): List of event mentions, each with the
                following keys:
                - "id" (str): Unique identifier for the event.
                - "event_type" (str): Type of the event.
                - "trigger" (dict): Trigger information with keys
                    - "text" (str)
                    - "start" (int)
                    - "end" (int)
                - "arguments" (list[dict]): List of arguments, each with keys
                    - "entity_id" (str)
                    - "role" (str)
                    - "text" (str)
                    - "start" (int)
                    - "end" (int)
            - "entity_mentions" (list[dict]): List of entity mentions, each with
                the following keys:
                - "id" (str): Unique identifier for the entity.
                - "text" (str): Text of the entity.
                - "entity_type" (str): Type of the entity.
                - "start" (int): Start index of the entity.
                - "end" (int): End index of the entity.
            - "lang" (str): Language of the document. Always 'en'.
    """
    data = []
    for obj in tqdm(objs, ncols=100):
        tokens = []
        for s in obj["sentences"]:
            tokens.extend(s)

        # This line of code was included in TextEE, but token_lens is never used
        # token_lens = [len(x) for x in tokens]

        entity_by_index = {}
        for k, entity in enumerate(obj["ent_spans"]):
            start = entity[0]
            end = entity[1] + 1
            entity_ = {
                "id": f"{obj['doc_key']}_Ent{k}",
                "text": " ".join(tokens[start:end]),
                "entity_type": "Entity",
                "start": start,
                "end": end,
            }
            entity_by_index[(start, end)] = entity_

        event_by_index = {}
        for k, event in enumerate(obj["evt_triggers"]):
            start = event[0]
            end = event[1] + 1
            event_type = event[2][0][0]

            event_dict = {
                "id": f"{obj['doc_key']}_Evt{k}",
                "event_type": event_type,
                "trigger": {
                    "text": " ".join(tokens[start:end]),
                    "start": start,
                    "end": end,
                },
                "arguments": [],
            }
            event_by_index[(start, end)] = event_dict

        for argument in obj["gold_evt_links"]:
            trigger = (argument[0][0], argument[0][1] + 1)

            start = argument[1][0]
            end = argument[1][1] + 1

            argument_ = {
                "entity_id": entity_by_index[(start, end)]["id"],
                "role": argument[2].split("arg", maxsplit=1)[-1][2:],
                "text": " ".join(tokens[start:end]),
                "start": start,
                "end": end,
            }
            event_by_index[trigger]["arguments"].append(argument_)

        entity_mentions = sorted(
            entity_by_index.values(), key=lambda x: (x["start"], x["end"])
        )
        event_mentions = sorted(
            event_by_index.values(),
            key=lambda x: (x["trigger"]["start"], x["trigger"]["end"]),
        )
        for event in event_mentions:
            event["arguments"].sort(key=lambda x: (x["start"], x["end"]))

        dt = {
            "doc_id": obj["doc_key"],
            "wnd_id": f"{obj['doc_key']}_1",
            "text": " ".join(tokens),
            "tokens": tokens,
            "event_mentions": event_mentions,
            "entity_mentions": entity_mentions,
            "lang": "en",
        }
        data.append(dt)

    return data


def process_rams_data(
    in_folder: str | PosixPath,
    out_folder: str | PosixPath,
    split_folder: str | PosixPath,
) -> None:
    """
    Processes RAMS data by reading documents, splitting them into train, dev, and
    test sets, converting them to the RAMS format, printing statistics for each set,
    and saving the processed data.

    The data is saved as JSON lines files in the TextEE format.

    Args:
        in_folder (str): The input folder containing the raw data documents.
        out_folder (str): The output folder where the processed data will be saved.
        split_folder (str): The folder containing the split information for
            train, dev, and test sets.
    """

    objs = read_docs(in_folder)

    train_objs, dev_objs, test_objs = get_split(objs, split_folder, key_name="doc_key")

    train_data = convert_rams_format(train_objs)
    dev_data = convert_rams_format(dev_objs)
    test_data = convert_rams_format(test_objs)

    print("Train")
    print_statistics(train_data)
    print("Dev")
    print_statistics(dev_data)
    print("Test")
    print_statistics(test_data)

    save_data(out_folder, train_data, dev_data, test_data)


def convert_wikievents_format(objs: list[dict], seg_map: dict) -> list[dict]:
    """
    Convert a list of event objects from the WikiEvents to the TextEE format.

    Args:
        objs (list[dict]): A list of dictionaries, each representing a document with
            the following keys:
            - "doc_id" (str): Unique identifier for the document.
            - "tokens" (list[str]): List of tokens in the document.
            - "sentences" (list[list[list[list[str, int, int]], str]]):
                List of chunks of text, where each text is a list of tokens, and two
                integers representing the start and end indices of the character span.
                Along with the list there is the sentence written out.
            - "entity_mentions" (list[dict]): List of entity mentions, each with the
                following keys:
                - "id" (str): Unique identifier for the entity.
                - "sent_idx" (int): Index of the sentence where the entity mention is.
                - "start" (int): Start index of the entity mention.
                - "end" (int): End index of the entity mention.
                - "entity_type" (str): Type of the entity.
                - "mention_type" (str): Type of the mention.
                - "text" (str): Text of the entity mention.
            - "relation_mentions" (list[]): List of relation mentions.
            - "event_mentions" (list[dict]): List of event mentions, each with the
                following keys:
                - "id" (str): Unique identifier for the event.
                - "event_type" (str): Type of the event.
                - "trigger" (dict): Trigger information with keys
                    - "text" (str): Text of the trigger.
                    - "sent_idx" (int): Index of the sentence where the trigger is.
                    - "start" (int): Start index of the trigger.
                    - "end" (int): End index of the trigger.
                - "arguments" (list[dict]): List of arguments, each with keys
                    - "entity_id" (str)
                    - "role" (str)
                    - "text" (str)
        seg_map (dict[str, list[list[int, int]]): A dictionary mapping document
            IDs to segment indices.

    Returns:
        list[dict]: A list of dictionaries, each representing a document in the
            TextEE format with the following keys:
            - "doc_id" (str): Unique identifier for the document.
            - "wnd_id" (str): Window identifier for the document.
            - "text" (str): Full text of the document.
            - "tokens" (list[str]): List of tokens in the document.
            - "event_mentions" (list[dict]): List of event mentions, each with the
                following keys:
                - "id" (str): Unique identifier for the event.
                - "event_type" (str): Type of the event.
                - "trigger" (dict): Trigger information with keys
                    - "text" (str): Text of the trigger.
                    - "start" (int): Index of the trigger in the document.
                    - "end" (int): Index of the trigger in the document.
                - "arguments" (list[dict]): List of arguments, each with keys
                    - "entity_id" (str): Unique identifier for the entity.
                    - "role" (str): Role of the argument.
                    - "text" (str): Text of the argument.
                    - "start" (int): Start index of the argument.
                    - "end" (int): End index of the argument.
            - "entity_mentions" (list[dict]): List of entity mentions, each with
                the following keys:
                - "id" (str): Unique identifier for the entity.
                - "text" (str): Text of the entity.
                - "entity_type" (str): Type of the entity.
                - "start" (int): Start index of the entity.
                - "end" (int): End index of the entity.
            - "lang" (str): Language of the document. Always 'en'.
    """
    data = []
    n_arg_drop = 0
    for obj in tqdm(objs, ncols=100):
        n_sent = len(obj["sentences"])
        # Assigned but never used:
        # sent_tokens = [[t[0] for t in s[0]] for s in obj["sentences"]]
        sent_lens = [len(s[0]) for s in obj["sentences"]]
        assert sum(sent_lens) == len(obj["tokens"])

        sent_token_start = [0] + [sum(sent_lens[: i + 1]) for i in range(n_sent)]

        sents = seg_map[obj["doc_id"]]

        n_seg = len(sents)
        sent_starts = [sum(sent_lens[0 : s[0]]) for s in sents]

        idx2seg_map = np.zeros((n_sent,), dtype=int)
        for i in range(n_sent):
            idx2seg_map[i] = -1
        for i, (idx1, idx2) in enumerate(sents):
            for j in range(idx1, idx2):
                idx2seg_map[j] = i

        entity_mentions = defaultdict(list)
        entity_map = {}
        for entity_mention in obj["entity_mentions"]:
            seg_loc = idx2seg_map[entity_mention["sent_idx"]]
            if seg_loc == -1:
                continue
            entity = {
                "id": entity_mention["id"],
                "text": entity_mention["text"],
                "entity_type": entity_mention["entity_type"],
                "start": entity_mention["start"] - sent_starts[seg_loc],
                "end": entity_mention["end"] - sent_starts[seg_loc],
            }
            entity_map[entity_mention["id"]] = (seg_loc, entity)
            entity_mentions[seg_loc].append(entity)

        event_mentions = defaultdict(list)
        for event_mention in obj["event_mentions"]:
            seg_loc = idx2seg_map[event_mention["trigger"]["sent_idx"]]
            if seg_loc == -1:
                continue

            arguments = []
            for arg in event_mention["arguments"]:
                entity_loc, entity = entity_map[arg["entity_id"]]
                if seg_loc != entity_loc:
                    n_arg_drop += 1
                    continue

                argument = {
                    "entity_id": arg["entity_id"],
                    "role": arg["role"],
                    "text": arg["text"],
                    "start": entity["start"],
                    "end": entity["end"],
                }
                arguments.append(argument)
            arguments.sort(key=lambda x: (x["start"], x["end"]))

            event = {
                "id": event_mention["id"],
                "event_type": event_mention["event_type"],
                "trigger": {
                    "text": event_mention["trigger"]["text"],
                    "start": event_mention["trigger"]["start"] - sent_starts[seg_loc],
                    "end": event_mention["trigger"]["end"] - sent_starts[seg_loc],
                },
                "arguments": arguments,
            }
            event_mentions[seg_loc].append(event)

        for i in range(n_seg):
            entity_mentions_ = entity_mentions[i]
            event_mentions_ = event_mentions[i]

            entity_mentions_ = sorted(
                entity_mentions_, key=lambda x: (x["start"], x["end"])
            )
            event_mentions_ = sorted(
                event_mentions_,
                key=lambda x: (x["trigger"]["start"], x["trigger"]["end"]),
            )
            dt = {
                "doc_id": obj["doc_id"],
                "wnd_id": f"{obj['doc_id']}_{i}",
                "text": " ".join(
                    obj["tokens"][
                        sent_token_start[sents[i][0]] : sent_token_start[sents[i][1]]
                    ]
                ),
                "tokens": obj["tokens"][
                    sent_token_start[sents[i][0]] : sent_token_start[sents[i][1]]
                ],
                "event_mentions": event_mentions_,
                "entity_mentions": entity_mentions_,
                "lang": "en",
            }
            data.append(dt)

    print(f"Number of dropped arguments: {n_arg_drop}")

    return data


def process_wikievents_data(
    in_folder: str | PosixPath,
    out_folder: str | PosixPath,
    split_folder: str | PosixPath,
    seg_map: str | PosixPath,
) -> None:
    """
    Processes WikiEvents data by reading documents, splitting them into train, dev, and
    test sets, converting them to the WikiEvents format, printing statistics for each
    set, and saving the processed data.
    This uses the segment map provided to convert the data to the TextEE format.

    The data is saved as JSON lines files in the TextEE format.

    Args:
        in_folder (str): The input folder containing the raw data documents.
        out_folder (str): The output folder where the processed data will be saved.
        split_folder (str): The folder containing the split information for
            train, dev, and test sets.
        seg_map (str): The path to the segment map JSON file.
    """
    with open(seg_map) as fp:
        s_map = json.load(fp)

    objs = read_docs(in_folder)

    train_objs, dev_objs, test_objs = get_split(objs, split_folder)

    train_data = convert_wikievents_format(train_objs, s_map)
    dev_data = convert_wikievents_format(dev_objs, s_map)
    test_data = convert_wikievents_format(test_objs, s_map)

    print("Train")
    print_statistics(train_data)
    print("Dev")
    print_statistics(dev_data)
    print("Test")
    print_statistics(test_data)

    save_data(out_folder, train_data, dev_data, test_data)
