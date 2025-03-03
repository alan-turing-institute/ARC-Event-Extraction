from __future__ import annotations

from arcee.stats.gold_compare import (
    load_events,
    stats_on_argument_detection,
    stats_on_event_detection,
)


def test_load_events_event_detection(event_trigger_jsonl_file):
    events = load_events(event_trigger_jsonl_file, "wnd_id")
    assert events is not None
    assert len(events) == 4
    assert events["p1"] == []
    assert events["p2"] == [{"Life:Marry": "married"}, {"Life:Injure": "wounded"}]
    assert events["p3"] == [{"Movement:Transport": "transported"}]
    assert events["p4"] == [{"Job:Hire": "hired"}]


def test_load_events_argument_extraction(event_arg_jsonl_file):
    events = load_events(event_arg_jsonl_file, "evt_id")
    assert events is not None
    assert len(events) == 4
    assert events["e1"] == {"Person": "Alice", "Place": "Las Vegas"}
    assert events["e2"] == {"Person": "Bob", "Vehicle": "car"}
    assert events["e3"] == {"Person": "Charlie", "Place": "New York"}
    assert events["e4"] == {"Person": "Dave", "House": None}


def test_stats_on_event_detection(event_trigger_jsonl_file):
    gold_events = load_events(event_trigger_jsonl_file, "wnd_id")

    perfect_stats = stats_on_event_detection(gold_events, gold_events)

    assert perfect_stats["total_passage"] == 4
    assert perfect_stats["correct_passage"] == 4
    assert perfect_stats["passage_accuracy"] == 1.0
    assert perfect_stats["total_event"] == 4
    assert perfect_stats["predict_event"] == 4
    assert perfect_stats["correct_all"] == 4
    assert perfect_stats["event_precision"] == 1.0
    assert perfect_stats["event_recall"] == 1.0
    assert perfect_stats["event_f1"] == 1.0
    assert perfect_stats["event_type_precision"] == 1.0
    assert perfect_stats["event_type_recall"] == 1.0
    assert perfect_stats["event_type_f1"] == 1.0

    output_with_mistakes = {
        "p1": [],
        "p2": [{"Life:Marry": "married"}, {"Life:Injure": "injured"}],  # Mistake here
        "p3": [{"Movement:Transport": "transported"}],
        "p4": [],  # Missing event here
    }
    mistake_stats = stats_on_event_detection(gold_events, output_with_mistakes)

    assert mistake_stats["total_passage"] == 4
    assert mistake_stats["correct_passage"] == 2  # two passages are incorrect
    assert mistake_stats["passage_accuracy"] == 2 / 4
    assert mistake_stats["total_event"] == 4
    assert mistake_stats["predict_event"] == 3
    assert mistake_stats["correct_all"] == 2  # Two events are incorrect
    assert mistake_stats["event_precision"] == 2 / 3
    assert mistake_stats["event_recall"] == 2 / 4
    assert mistake_stats["event_f1"] == 2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4))
    assert mistake_stats["event_type_precision"] == 3 / 3
    assert mistake_stats["event_type_recall"] == 3 / 4
    assert mistake_stats["event_type_f1"] == 2 * (3 / 3) * (3 / 4) / ((3 / 3) + (3 / 4))


def test_stats_on_argument_detection(event_arg_jsonl_file):
    gold_events = load_events(event_arg_jsonl_file, "evt_id")

    perfect_stats = stats_on_argument_detection(gold_events, gold_events)

    assert perfect_stats["total_event"] == 4
    assert perfect_stats["correct_event"] == 4
    assert perfect_stats["event_accuracy"] == 1.0
    assert perfect_stats["total_argument"] == 7
    assert perfect_stats["predict_argument"] == 7
    assert perfect_stats["correct_argument"] == 7
    assert perfect_stats["argument_precision"] == 1.0
    assert perfect_stats["argument_recall"] == 1.0
    assert perfect_stats["argument_f1"] == 1.0

    output_with_mistakes = {
        "e1": {"Person": "Alice", "Place": "Las Vegas"},
        "e2": {"Person": "Bob", "Vehicle": "bike"},  # Mistake here
        "e3": {"Person": "Charlie"},  # Missing argument here
        "e4": {"Person": "Dave", "House": "apartment"},
    }
    mistake_stats = stats_on_argument_detection(gold_events, output_with_mistakes)

    assert mistake_stats["total_event"] == 4
    assert mistake_stats["correct_event"] == 1  # Two events are incorrect
    assert mistake_stats["event_accuracy"] == 1 / 4
    assert mistake_stats["total_argument"] == 7
    assert mistake_stats["predict_argument"] == 7
    assert mistake_stats["correct_argument"] == 5  # Two arguments are incorrect
    assert mistake_stats["argument_precision"] == 5 / 7
    assert mistake_stats["argument_recall"] == 5 / 7
    assert mistake_stats["argument_f1"] == 2 * (5 / 7) * (5 / 7) / ((5 / 7) + (5 / 7))
