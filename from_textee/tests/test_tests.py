from __future__ import annotations

import json
import os
from pathlib import PosixPath


def test_splits(split_data_paths: list[PosixPath]) -> None:
    # This checks that the fake data made to check the RMAS scripts were correct
    # Check if the split files were created
    assert all(os.path.exists(path) for path in split_data_paths)
    # Check if the split files have the expected content
    for path in split_data_paths:
        with open(path) as f:
            for line in f.readlines():
                assert line[:4] == "doc_"
                assert int(line[4:]) in range(10)


def test_rams_data(
    rams_data_paths: list[PosixPath], events_to_test: list[dict[str, str]]
) -> None:
    # This checks that the fake dta made to check the RMAS scripts were correct
    t_events = {f"doc_{i}": event for i, event in enumerate(events_to_test)}
    # Check if the data files were created
    assert all(os.path.exists(path) for path in rams_data_paths)
    # Check if the data files have the expected content
    for path in rams_data_paths:
        with open(path) as f:
            for line in f.readlines():
                json_line = json.loads(line)
                assert json_line["doc_key"][:4] == "doc_"
                assert json_line["doc_key"] in t_events
                assert (
                    json_line["sentences"][0][1]
                    == t_events[json_line["doc_key"]]["Sentence"].split(" ")[1]
                )
                assert (
                    json_line["evt_triggers"][0][2][0][0]
                    == t_events[json_line["doc_key"]]["EventType"]
                )


def test_wikievent_data(
    wikievents_data_paths: list[PosixPath], events_to_test: list[dict[str, str]]
) -> None:
    # This checks that the fake data made to check the WikiEvents scripts were correct
    t_events = {f"doc_{i}": event for i, event in enumerate(events_to_test)}
    # Check if the data files were created
    assert all(os.path.exists(path) for path in wikievents_data_paths)
    # Check if the data files have the expected content
    for path in wikievents_data_paths:
        with open(path) as f:
            for line in f.readlines():
                json_line = json.loads(line)
                assert json_line["doc_id"][:4] == "doc_"
                assert json_line["doc_id"] in t_events
                assert (
                    json_line["tokens"][1]
                    == t_events[json_line["doc_id"]]["Sentence"].split(" ")[1]
                )
                assert (
                    json_line["event_mentions"][0]["trigger"]["text"]
                    == t_events[json_line["doc_id"]]["EventTrigger"]
                )
                assert (
                    json_line["event_mentions"][0]["event_type"]
                    == t_events[json_line["doc_id"]]["EventType"]
                )


def test_wikievents_seg_path(wikievents_seg_path: PosixPath) -> None:
    # This checks that the fake data made to check the WikiEvents scripts were correct
    # Check if the segment map file was created
    assert os.path.exists(wikievents_seg_path)
    with open(wikievents_seg_path) as f:
        seg_map = json.load(f)

    assert isinstance(seg_map, dict)
    assert isinstance(seg_map["doc_0"], list)
    assert seg_map["doc_0"][0][0] == 0
    assert isinstance(seg_map["doc_0"][0][1], int)
