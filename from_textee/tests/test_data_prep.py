from __future__ import annotations

import json
import os
from pathlib import PosixPath

import pytest

from from_textee import data_prep


@pytest.mark.usefixtures("rams_data_paths")
def test_read_docs(testing_rams_dir: PosixPath) -> None:
    all_objs = data_prep.read_docs(testing_rams_dir)

    assert len(all_objs) == 10
    assert isinstance(all_objs, list)
    assert isinstance(all_objs[0], dict)


def test_save_data(
    save_testing_data_dir: PosixPath,
    events_to_test: list[dict[str, str]],
) -> None:
    # Get the data to save
    train_data = events_to_test[:6]
    dev_data = events_to_test[6:8]
    test_data = events_to_test[8:]
    # Save and load data
    data_prep.save_data(save_testing_data_dir, train_data, dev_data, test_data)
    all_objs = data_prep.read_docs(save_testing_data_dir)
    # Check the data is what is expected
    assert len(all_objs) == 10
    assert isinstance(all_objs, list)
    assert isinstance(all_objs[0], dict)


@pytest.mark.usefixtures("split_data_paths")
def test_get_split(
    testing_split_dir: PosixPath,
    events_to_test: list[dict[str, str]],
) -> None:
    t_events = events_to_test.copy()
    for i, event in enumerate(t_events):
        event["doc_id"] = f"doc_{i}"
    train_objs, dev_objs, test_objs = data_prep.get_split(t_events, testing_split_dir)
    assert isinstance(train_objs, list)
    assert isinstance(train_objs[0], dict)
    assert all((len(train_objs) == 6, len(dev_objs) == 2, len(test_objs) == 2))
    assert train_objs[0]["doc_id"] == "doc_2"
    assert dev_objs[0]["doc_id"] == "doc_8"
    assert test_objs[0]["doc_id"] == "doc_0"


def data_format_tester(
    all_objs: list[dict], events_to_test: list[dict[str, str]]
) -> None:
    t_events = {f"doc_{i}": event for i, event in enumerate(events_to_test)}
    assert isinstance(all_objs, list)
    assert isinstance(all_objs[0], dict)
    for obj in all_objs:
        assert obj["doc_id"] in t_events
        assert obj["lang"] == "en"
        assert obj["text"][:20] == t_events[obj["doc_id"]]["Sentence"][:20]
        assert (
            obj["event_mentions"][0]["event_type"]
            == t_events[obj["doc_id"]]["EventType"]
        )
        assert (
            obj["event_mentions"][0]["trigger"]["text"]
            == t_events[obj["doc_id"]]["EventTrigger"]
        )
        assert obj["event_mentions"][0]["trigger"]["start"] == t_events[obj["doc_id"]][
            "Sentence"
        ].split(" ").index(t_events[obj["doc_id"]]["EventTrigger"])
        assert obj["event_mentions"][0]["arguments"][0]["role"] == "Subject"
        assert obj["event_mentions"][0]["arguments"][1]["role"] == "Object"
        assert (
            obj["event_mentions"][0]["arguments"][0]["text"]
            == t_events[obj["doc_id"]]["Subject"]
        )
        assert (
            obj["event_mentions"][0]["arguments"][1]["text"]
            == t_events[obj["doc_id"]]["Object"]
        )


@pytest.mark.usefixtures("rams_data_paths")
def test_convert_rams_format(
    testing_rams_dir: PosixPath,
    events_to_test: list[dict[str, str]],
) -> None:
    all_objs = data_prep.read_docs(testing_rams_dir)
    new_objs = data_prep.convert_rams_format(all_objs)
    data_format_tester(new_objs, events_to_test)


@pytest.mark.usefixtures("rams_data_paths")
def test_rams_print_statistics(
    testing_rams_dir: PosixPath,
) -> None:
    all_objs = data_prep.read_docs(testing_rams_dir)
    new_objs = data_prep.convert_rams_format(all_objs)
    data_prep.print_statistics(new_objs)
    assert True  # Just checking nothing throws error


@pytest.mark.usefixtures("rams_data_paths", "split_data_paths")
def test_process_rams_data(
    testing_rams_dir: PosixPath,
    testing_split_dir: PosixPath,
    save_testing_data_dir: PosixPath,
) -> None:
    data_prep.process_rams_data(
        testing_rams_dir, save_testing_data_dir, testing_split_dir
    )
    assert all(
        os.path.exists(save_testing_data_dir / f"{file}.jsonlines")
        for file in ["train", "dev", "test"]
    )


@pytest.mark.usefixtures("wikievents_data_paths")
def test_convert_wikievents_format(
    testing_wikievents_dir: PosixPath,
    wikievents_seg_path: PosixPath,
    events_to_test: list[dict[str, str]],
) -> None:
    with open(wikievents_seg_path) as fseg:
        seg_map = json.load(fseg)
    all_objs = data_prep.read_docs(testing_wikievents_dir)
    new_objs = data_prep.convert_wikievents_format(all_objs, seg_map)
    data_format_tester(new_objs, events_to_test)


@pytest.mark.usefixtures("wikievents_data_paths")
def test_wikievents_print_statistics(
    testing_wikievents_dir: PosixPath,
    wikievents_seg_path: PosixPath,
) -> None:
    with open(wikievents_seg_path) as fseg:
        seg_map = json.load(fseg)
    all_objs = data_prep.read_docs(testing_wikievents_dir)
    new_objs = data_prep.convert_wikievents_format(all_objs, seg_map)
    data_prep.print_statistics(new_objs)
    assert True  # Just checking nothing throws error


@pytest.mark.usefixtures("wikievents_data_paths", "split_data_paths")
def test_process_wikievents_data(
    testing_wikievents_dir: PosixPath,
    testing_split_dir: PosixPath,
    save_testing_data_dir: PosixPath,
    wikievents_seg_path: PosixPath,
) -> None:
    data_prep.process_wikievents_data(
        testing_wikievents_dir,
        save_testing_data_dir,
        testing_split_dir,
        wikievents_seg_path,
    )
    assert all(
        os.path.exists(save_testing_data_dir / f"{file}.jsonlines")
        for file in ["train", "dev", "test"]
    )
