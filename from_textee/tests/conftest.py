from __future__ import annotations

import json

import pytest

some_events = [
    {
        "Sentence": "Linda dusted the living room furniture.",
        "EventType": "Clean",
        "EventTrigger": "dusted",
        "Subject": "Linda",
        "Object": "furniture",
    },
    {
        "Sentence": "Sophia boiled pasta and then tossed it with creamy alfredo sauce.",
        "EventType": "Cook",
        "EventTrigger": "boiled",
        "Subject": "Sophia",
        "Object": "pasta",
    },
    {
        "Sentence": "Emily embarked on a road trip to explore the national parks.",
        "EventType": "Travel",
        "EventTrigger": "embarked",
        "Subject": "Emily",
        "Object": "trip",
    },
    {
        "Sentence": "Sophia practised Pilates on a mat in her living room.",
        "EventType": "Exercise",
        "EventTrigger": "practised",
        "Subject": "Sophia",
        "Object": "Pilates",
    },
    {
        "Sentence": "Laura bumped into her college professor at the bookshop.",
        "EventType": "Meet",
        "EventTrigger": "bumped",
        "Subject": "Laura",
        "Object": "professor",
    },
    {
        "Sentence": "The cyclist dislocated his shoulder after a collision with a car.",
        "EventType": "Injure",
        "EventTrigger": "dislocated",
        "Subject": "cyclist",
        "Object": "shoulder",
    },
    {
        "Sentence": "Anna selected a bouquet from the florist for her friend.",
        "EventType": "Purchase",
        "EventTrigger": "selected",
        "Subject": "Anna",
        "Object": "bouquet",
    },
    {
        "Sentence": "The airline transported cargo to the international terminal.",
        "EventType": "Transport",
        "EventTrigger": "transported",
        "Subject": "airline",
        "Object": "cargo",
    },
    {
        "Sentence": "The university hired Dr. Smith to lead the biology department.",
        "EventType": "Hire",
        "EventTrigger": "hired",
        "Subject": "university",
        "Object": "Smith",
    },
    {
        "Sentence": "The architect designed and watched the construction of a house.",
        "EventType": "Build",
        "EventTrigger": "construction",
        "Subject": "architect",
        "Object": "house",
    },
]


@pytest.fixture(scope="module")
def events_to_test() -> list[dict[str, str]]:
    # Return the fake events to test
    return some_events


@pytest.fixture(scope="module")
def testing_rams_dir(tmp_path_factory: pytest.fixture.tmp_path_factory) -> str:
    # Create a temporary directory
    return tmp_path_factory.mktemp("data_rams")


@pytest.fixture(scope="module")
def rams_data_paths(testing_rams_dir: pytest.fixture) -> str:
    # Build the RAMS data in the correct format
    rams_jsons = []
    for i, event in enumerate(some_events):
        new_event = {"doc_key": f"doc_{i}"}
        new_event["sentences"] = [event["Sentence"].replace(".", " .").split(" ")]
        trig_loc = new_event["sentences"][0].index(event["EventTrigger"])
        subj_loc = new_event["sentences"][0].index(event["Subject"])
        obj_loc = new_event["sentences"][0].index(event["Object"])
        new_event["evt_triggers"] = [[trig_loc, trig_loc, [event["EventType"], 1.0]]]
        new_event["ent_spans"] = [
            [subj_loc, subj_loc, ["Subject", 1.0]],
            [obj_loc, obj_loc, ["Object", 1.0]],
        ]
        new_event["gold_evt_links"] = [
            [[trig_loc, trig_loc], [subj_loc, subj_loc], "Subject"],
            [[trig_loc, trig_loc], [obj_loc, obj_loc], "Object"],
        ]
        rams_jsons.append(new_event)

    # Write the RAMS data to files
    file_names = [
        ("train.jsonlines", rams_jsons[:6]),
        ("dev.jsonlines", rams_jsons[6:8]),
        ("test.jsonlines", rams_jsons[8:]),
    ]
    file_paths = []
    for file_name, data in file_names:
        file_path = testing_rams_dir / file_name
        with open(file_path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")
        file_paths.append(file_path)

    return file_paths


@pytest.fixture(scope="module")
def testing_split_dir(tmp_path_factory: pytest.fixture.tmp_path_factory) -> str:
    # Create a temporary directory
    return tmp_path_factory.mktemp("split")


@pytest.fixture(scope="module")
def split_data_paths(testing_split_dir: pytest.fixture) -> tuple[str, str, str]:
    # This builds the split data in the correct format
    keys = [f"doc_{i}" for i in range(10)]
    files = [
        ("train.txt", keys[2:8]),
        ("dev.txt", keys[8:]),
        ("test.txt", keys[:2]),
    ]
    file_paths = []
    for file, data in files:
        file_path = testing_split_dir / file
        with open(file_path, "w") as f:
            for key in data:
                f.write(key + "\n")
        file_paths.append(file_path)

    return file_paths
