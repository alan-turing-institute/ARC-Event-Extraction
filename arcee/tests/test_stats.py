from __future__ import annotations

from arcee.stats.gold_compare import (
    compare_argument_detection,
    compare_events_detection,
    load_events,
    semi_compare_argument_detection,
    semi_compare_events_detection,
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


def test_compare_events_detection():
    gold_events = [
        {"Life:Marry": "married"},
        {"Life:Injure": "wounded"},
    ]
    correct_passage, correct_all, correct_type = compare_events_detection(
        gold_events, gold_events
    )
    assert correct_passage == 1
    assert correct_all == 2
    assert correct_type == 2

    output_events_incorrect = [
        {"Life:Marry": "married"},
        {"Life:Injure": "injured"},
    ]
    correct_passage, correct_all, correct_type = compare_events_detection(
        gold_events, output_events_incorrect
    )
    assert correct_passage == 0
    assert correct_all == 1
    assert correct_type == 2

    output_events_partial = [
        {"Life:Marry": "married"},
    ]
    correct_passage, correct_all, correct_type = compare_events_detection(
        gold_events, output_events_partial
    )
    assert correct_passage == 0
    assert correct_all == 1
    assert correct_type == 1

    null_passage, null_all, null_type = compare_events_detection([], [])
    assert null_passage == 1
    assert null_all == 0
    assert null_type == 0


def test_semi_compare_events_detection():
    gold_events = [
        {"Life:Marry": "married"},
        {"Life:Injure": "wounded"},
    ]

    correct_passage, correct_all, correct_type = semi_compare_events_detection(
        gold_events, gold_events
    )
    assert correct_passage == 1
    assert correct_all == 2
    assert correct_type == 2

    output_events_partial_match = [
        {"Life:Marry": "married"},
        {"Life:Injure": "wound"},
    ]
    correct_passage, correct_all, correct_type = semi_compare_events_detection(
        gold_events, output_events_partial_match
    )
    assert correct_passage == 1
    assert correct_all == 2
    assert correct_type == 2

    output_events_extra_words = [
        {"Life:Marry": "married"},
        {"Life:Injure": "badly wounded"},
    ]
    correct_passage, correct_all, correct_type = semi_compare_events_detection(
        gold_events, output_events_extra_words
    )
    assert correct_passage == 1
    assert correct_all == 2
    assert correct_type == 2

    output_events_incorrect = [
        {"Life:Marry": "married"},
        {"Life:Injure": "injured"},
    ]
    correct_passage, correct_all, correct_type = semi_compare_events_detection(
        gold_events, output_events_incorrect
    )
    assert correct_passage == 0
    assert correct_all == 1
    assert correct_type == 2

    output_events_partial = [
        {"Life:Marry": "married"},
    ]
    correct_passage, correct_all, correct_type = semi_compare_events_detection(
        gold_events, output_events_partial
    )
    assert correct_passage == 0
    assert correct_all == 1
    assert correct_type == 1

    null_passage, null_all, null_type = semi_compare_events_detection([], [])
    assert null_passage == 1
    assert null_all == 0
    assert null_type == 0


def test_compare_argument_detection():
    gold_arguments = {
        "Person": "Alice",
        "Place": "Las Vegas",
    }
    correct_event, correct_argument = compare_argument_detection(
        gold_arguments, gold_arguments
    )
    assert correct_event == 1
    assert correct_argument == 2

    output_arguments_incorrect = {
        "Person": "Alice",
        "Place": "New York",
    }
    correct_event, correct_argument = compare_argument_detection(
        gold_arguments, output_arguments_incorrect
    )
    assert correct_event == 0
    assert correct_argument == 1

    output_arguments_extra = {
        "Person": "Alice",
        "Place": "Las Vegas",
        "Vehicle": "car",
    }
    correct_event, correct_argument = compare_argument_detection(
        gold_arguments, output_arguments_extra
    )
    assert correct_event == 0
    assert correct_argument == 2

    output_arguments_partial = {
        "Person": "Alice",
    }
    correct_event, correct_argument = compare_argument_detection(
        gold_arguments, output_arguments_partial
    )
    assert correct_event == 0
    assert correct_argument == 1

    null_event, null_argument = compare_argument_detection({}, {})
    assert null_event == 1
    assert null_argument == 0


def test_semi_compare_argument_detection():
    gold_arguments = {
        "Person": "Alice",
        "Place": "Las Vegas",
    }
    correct_event, correct_argument = semi_compare_argument_detection(
        gold_arguments, gold_arguments
    )
    assert correct_event == 1
    assert correct_argument == 2

    output_arguments_partial_match = {
        "Person": "Alice",
        "Place": "Vegas",
    }
    correct_event, correct_argument = semi_compare_argument_detection(
        gold_arguments, output_arguments_partial_match
    )
    assert correct_event == 1
    assert correct_argument == 2

    output_arguments_extra_words = {
        "Person": "Alice",
        "Place": "Las Vegas, Nevada",
    }
    correct_event, correct_argument = semi_compare_argument_detection(
        gold_arguments, output_arguments_extra_words
    )
    assert correct_event == 1
    assert correct_argument == 2

    output_arguments_incorrect = {
        "Person": "Alice",
        "Place": "New York",
    }
    correct_event, correct_argument = semi_compare_argument_detection(
        gold_arguments, output_arguments_incorrect
    )
    assert correct_event == 0
    assert correct_argument == 1

    output_arguments_partial = {
        "Person": "Alice",
    }
    correct_event, correct_argument = semi_compare_argument_detection(
        gold_arguments, output_arguments_partial
    )
    assert correct_event == 0
    assert correct_argument == 1

    null_event, null_argument = semi_compare_argument_detection({}, {})
    assert null_event == 1
    assert null_argument == 0


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


def test_stats_on_event_detection_with_semi(event_trigger_jsonl_file):
    gold_events = load_events(event_trigger_jsonl_file, "wnd_id")

    perfect_stats = stats_on_event_detection(gold_events, gold_events, semi=True)

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
    mistake_stats = stats_on_event_detection(
        gold_events, output_with_mistakes, semi=True
    )

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

    output_with_overlap = {
        "p1": [],
        "p2": [{"Life:Marry": "married"}, {"Life:Injure": "badly wounded"}],
        "p3": [{"Movement:Transport": "transport"}],
        "p4": [{"Job:Hire": "fish"}],
    }

    overlap_stats = stats_on_event_detection(
        gold_events, output_with_overlap, semi=True
    )

    assert overlap_stats["total_passage"] == 4
    assert overlap_stats["correct_passage"] == 3  # one passage is incorrect
    assert overlap_stats["passage_accuracy"] == 3 / 4
    assert overlap_stats["total_event"] == 4
    assert overlap_stats["predict_event"] == 4
    assert overlap_stats["correct_all"] == 3  # One event is incorrect
    assert overlap_stats["event_precision"] == 3 / 4
    assert overlap_stats["event_recall"] == 3 / 4
    assert overlap_stats["event_f1"] == 2 * (3 / 4) * (3 / 4) / ((3 / 4) + (3 / 4))
    assert overlap_stats["event_type_precision"] == 4 / 4
    assert overlap_stats["event_type_recall"] == 4 / 4
    assert overlap_stats["event_type_f1"] == 2 * (4 / 4) * (4 / 4) / ((4 / 4) + (4 / 4))


def test_stats_on_argument_detection_with_semi(event_arg_jsonl_file):
    gold_events = load_events(event_arg_jsonl_file, "evt_id")

    perfect_stats = stats_on_argument_detection(gold_events, gold_events, semi=True)

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
    mistake_stats = stats_on_argument_detection(
        gold_events, output_with_mistakes, semi=True
    )

    assert mistake_stats["total_event"] == 4
    assert mistake_stats["correct_event"] == 1  # Two events are incorrect
    assert mistake_stats["event_accuracy"] == 1 / 4
    assert mistake_stats["total_argument"] == 7
    assert mistake_stats["predict_argument"] == 7
    assert mistake_stats["correct_argument"] == 5  # Two arguments are incorrect
    assert mistake_stats["argument_precision"] == 5 / 7
    assert mistake_stats["argument_recall"] == 5 / 7
    assert mistake_stats["argument_f1"] == 2 * (5 / 7) * (5 / 7) / ((5 / 7) + (5 / 7))

    output_with_overlap = {
        "e1": {"Person": "Alice", "Place": "Vegas"},
        "e2": {"Person": "Bob", "Vehicle": "car"},
        "e3": {"Person": "Charlie", "Place": "New York City"},
        "e4": {"Person": "Fred", "House": None},
    }

    overlap_stats = stats_on_argument_detection(
        gold_events, output_with_overlap, semi=True
    )

    assert overlap_stats["total_event"] == 4
    assert overlap_stats["correct_event"] == 3  # one event is incorrect
    assert overlap_stats["event_accuracy"] == 3 / 4
    assert overlap_stats["total_argument"] == 7
    assert overlap_stats["predict_argument"] == 7
    assert overlap_stats["correct_argument"] == 6  # one argument is incorrect
    assert overlap_stats["argument_precision"] == 6 / 7
    assert overlap_stats["argument_recall"] == 6 / 7
    assert overlap_stats["argument_f1"] == 2 * (6 / 7) * (6 / 7) / ((6 / 7) + (6 / 7))
