from __future__ import annotations

import json


def load_events(file: str, key: str) -> dict[str, dict | list] | None:
    """
    Load event data from a file and return a dictionary with the key as the key and
    the 'output' as the value.

    This is to use

    Args:
        file (str): The file to load the event data from.
        key (str): The key to use as the key in the dictionary. This should either be
            'evt_id' or 'wnd_id'.

    Returns:
        all_event (dict[str, dict | list] | None):
            This is a dictionary with the key as the key and the 'output' as the value.
            If the key is not found in the data, then this will return None.
    """
    with open(file) as g_file:
        first_line = g_file.readline()
        data = json.loads(first_line)
        if key not in data:
            return None

    all_events = {}
    with open(file) as g_file:
        for line in g_file:
            data = json.loads(line)
            all_events[data[key]] = data["output"]
    return all_events


class SemiString(str):
    """
    This is a string that is considered equal when either string contains a subset of
    the other.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, str | SemiString):
            return NotImplemented
        return (self in other) or (other in self)


def _dict_to_semi_dict(d: dict[str, str | None]) -> dict[str, SemiString | None]:
    """
    Convert a dictionary to a dictionary with SemiString values.

    Args:
        d (dict[str, str | None]): The dictionary to convert.

    Returns:
        dict[str, SemiString | None]: The dictionary with SemiString values.
    """
    return {key: SemiString(val) if val is not None else None for key, val in d.items()}


def compare_events_detection(g_events: list, o_events: list) -> tuple[int, int, int]:
    """
    Compare gold events with output events and return the count of correct events and
    correct event types for a single passage.
    This does an exact sting match for the trigger.

    Args:
        g_events (list): Gold standard events detection result.
        o_events (list): Output events detection result.

    Returns:
        correct_passage (int): If the whole passage is correct.
        correct_all (int): The total number of correct events in the output data.
        correct_type (int): The total number events whose type was correctly predicted.
    """
    correct_all = 0
    correct_type = 0

    correct_passage = 1 if (g_events == o_events) else 0

    if len(g_events) == 0 or len(o_events) == 0:
        return correct_passage, correct_all, correct_type

    for g_event in g_events:
        event_correct = False
        event_type_correct = False
        for o_event in o_events:
            if g_event == o_event:
                event_correct = True
            if g_event.keys() == o_event.keys():
                event_type_correct = True

        if event_correct:
            correct_all += 1
        if event_type_correct:
            correct_type += 1

    return correct_passage, correct_all, correct_type


def semi_compare_events_detection(
    g_events: list[dict], o_events: list[dict]
) -> tuple[int, int, int]:
    """
    Compare gold events with output events and return the count of correct events and
    correct event types for a single passage.
    This does a partial string match for the trigger.

    Args:
        g_events (list): Gold standard events detection result.
        o_events (list): Output events detection result.

    Returns:
        correct_passage (int): If the whole passage is correct.
        correct_all (int): The total number of correct events in the output data.
        correct_type (int): The total number events whose type was correctly predicted.
    """
    semi_g_events = [_dict_to_semi_dict(event) for event in g_events]
    semi_o_events = [_dict_to_semi_dict(event) for event in o_events]

    return compare_events_detection(semi_g_events, semi_o_events)


def remove_event_argument_whitespace(
    events: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, str | None]]:
    """
    This function removes the whitespace from the arguments in the events.
    This means " bob freeman" and "bob freeman" will both be converted to "bobfreeman"
    before comparison.

    Args:
        events (dict[str, dict[str, str]]): The events to remove the whitespace from.

    Returns:
        whitespace_removed (dict[str, dict[str, str]]): The events with the whitespace
            removed.
    """

    whitespace_removed: dict[str, dict[str, str | None]] = {}
    for key, event in events.items():
        whitespace_removed[key] = {
            arg_type: arg.replace(" ", "") if arg else None
            for arg_type, arg in event.items()
        }

    return whitespace_removed


def remove_event_detection_whitespace(
    passages: dict[str, dict[str, str] | list[dict[str, str]]],
) -> dict[str, dict[str, str] | list[dict[str, str]]]:
    """
    A function to remove the whitespace from the event triggers in the passage.
    This means " bob freeman" and "bob freeman" will both be converted to "bobfreeman"
    before comparison.

    Args:
        passages (dict[str, list[dict[str, str]] | None]): The passage to remove the
            whitespace from.

    Returns:
        whitespace_removed (dict[str, list[dict[str, str]] | dict | None ]): The
            passage with the whitespace removed.
    """
    whitespace_removed: dict[str, dict[str, str] | list[dict[str, str]]] = {}
    for key, passage in passages.items():
        if isinstance(passage, dict):
            whitespace_removed[key] = passage
            continue
        if isinstance(passage, list):
            whitespace_removed[key] = [
                {
                    evt_type: evt_tig.replace(" ", "")
                    for evt_type, evt_tig in event.items()
                }
                for event in passage
            ]

    return whitespace_removed


def compare_argument_detection(g_args: dict, o_args: dict) -> tuple[int, int]:
    """
    Compare gold arguments with output arguments and return the count of correct events,
    correct arguments, and correct argument types for a single event.

    Args:
        g_args (dict): Gold standard arguments detection result.
        o_args (dict): Output arguments detection result.

    Returns:
        correct_event (int): If the whole event is correct.
        correct_argument (int): The total number of correct arguments.
    """
    correct_event = 1 if g_args == o_args else 0
    correct_argument = 0

    for arg_type, arg in g_args.items():
        if arg and arg_type in o_args:
            correct_argument += 1 if o_args[arg_type] == arg else 0

    return correct_event, correct_argument


def semi_compare_argument_detection(g_args: dict, o_args: dict) -> tuple[int, int]:
    """
    Compare gold arguments with output arguments and return the count of correct events,
    correct arguments, and correct argument types for a single event.
    This does a partial string match for the arguments.

    Args:
        g_args (dict): Gold standard arguments detection result.
        o_args (dict): Output arguments detection result.

    Returns:
        correct_event (int): If the whole event is correct.
        correct_argument (int): The total number of correct arguments.
    """
    semi_g_args = _dict_to_semi_dict(g_args)
    semi_o_args = _dict_to_semi_dict(o_args)

    return compare_argument_detection(semi_g_args, semi_o_args)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    """
    Safely divide two numbers, returning 0 if the denominator is 0.

    Args:
        numerator (int | float): The numerator.
        denominator (int | float): The denominator.

    Returns:
        float: The result of the division, or 0 if the denominator is 0.
    """
    return numerator / denominator if denominator != 0 else 0.0


def stats_on_event_detection(
    golds: dict[str, list | dict],
    outputs: dict[str, list | dict],
    semi: bool = False,
    real_gold: bool = True,
    keep_whitespace: bool = False,
) -> dict[str, int | float]:
    """
    Calculate the statistics on the event detection task.

    For the event type it calculates if an real event is included in the listed events.
    This means for passages with multiple events, it will count as correct if at least
    one event is of the correct type.

    The last field `event_f1` is the relevant F1 score to compare with TextEE.

    Args:
        golds (dict[str, dict | list]): The gold data in the format of a dictionary with
            the key as the key and the 'output' as the value.
        outputs (dict[str, dict | list]): The output data in the format of a dictionary
            with the key as the key and the 'output' as the value.
        semi (bool, optional): If the partial string match should be used. Defaults to
            False.
        real_gold (bool, optional): If the gold data is real or generated so can
            contain mistakes. Defaults to True.
        keep_whitespace (bool, optional): If the whitespace should be kept in the
            comparison. Defaults to False.

    Returns:
        dict[str, int | float]: This is a dict with the following keys:
            - total_passage (int): The total number of passages.
            - passage_accuracy (float): The accuracy of the passage level.
            - correct_passage (int): The total number of correct passages.
            - total_event (int): The total number of events in the gold data.
            - predict_event (int): The total number of events in the output data.
            - correct_all (int): The total number of correct events in the output data.
            - event_type_precision (float): The precision of the event types.
            - event_type_recall (float): The recall of the event types.
            - event_type_f1 (float): The f1 score of the event types.
            - event_precision (float): The precision of the events.
            - event_recall (float): The recall of the events.
            - event_f1 (float): The f1 score of the events.
    """

    total_passage = 0
    correct_passage = 0
    total_event = 0
    predict_event = 0
    correct_type = 0
    correct_all = 0

    if not keep_whitespace:
        golds = remove_event_detection_whitespace(golds)
        outputs = remove_event_detection_whitespace(outputs)

    for key, output in outputs.items():
        if real_gold and key not in golds:
            err_msg = f"Key {key} not found in gold."
            raise ValueError(err_msg)

        if not real_gold and key not in golds:
            continue  # This checks that the key is in the gold data
        if not real_gold and "JSON_ERROR" in golds[key]:
            continue  # This checks that the gold data is a valid JSON

        total_passage += 1
        total_event += len(golds[key])

        if not isinstance(output, list):
            # Need the type checks as llm can misbehave
            continue

        predict_event += len(output)

        this_correct_passage, this_correct_all, this_correct_type = (
            compare_events_detection(golds[key], output)  # type: ignore[arg-type]
            if not semi
            else semi_compare_events_detection(golds[key], output)  # type: ignore[arg-type]
        )
        correct_passage += this_correct_passage
        correct_all += this_correct_all
        correct_type += this_correct_type

    passage_accuracy = safe_divide(correct_passage, total_passage)
    event_precision = safe_divide(correct_all, predict_event)
    event_recall = safe_divide(correct_all, total_event)
    event_f1 = safe_divide(
        2 * event_precision * event_recall, event_precision + event_recall
    )
    event_type_precision = safe_divide(correct_type, predict_event)
    event_type_recall = safe_divide(correct_type, total_event)
    event_type_f1 = safe_divide(
        2 * event_type_precision * event_type_recall,
        event_type_precision + event_type_recall,
    )

    return {
        "total_passage": total_passage,
        "correct_passage": correct_passage,
        "passage_accuracy": passage_accuracy,
        "total_event": total_event,
        "predict_event": predict_event,
        "correct_all": correct_all,
        "event_type_precision": event_type_precision,
        "event_type_recall": event_type_recall,
        "event_type_f1": event_type_f1,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
    }


def stats_on_argument_detection(
    golds: dict[str, dict],
    outputs: dict[str, dict],
    semi: bool = False,
    real_gold: bool = True,
    keep_whitespace: bool = False,
) -> dict[str, int | float]:
    """
    Calculate the statistics on the argument detection task.

    The last field `argument_f1` is the relevant F1 score to compare with TextEE.

    Args:
        golds (dict[str, dict | list]): The gold data in the format of a dictionary with
            the key as the key and the 'output' as the value.
        outputs (dict[str, dict | list]): The output data in the format of a dictionary
            with the key as the key and the 'output' as the value.
        semi (bool, optional): If the partial string match should be used. Defaults to
            False.
        real_gold (bool, optional): If the gold data is real or generated so can
            contain mistakes. Defaults to True.
        keep_whitespace (bool, optional): If the whitespace should be kept in the
            comparison. Defaults to False.

    Returns:
        dict[str, int | float]: This is a dict with the following values:
            - total_event (int): The total number of events in the gold data.
            - correct_event (int): The total number of correct events in the data.
            - event_accuracy (float): The accuracy of the events.
            - total_argument (int): The total number of arguments in the gold data.
            - predict_argument (int): The total number of arguments in the output data.
            - correct_argument (int): The total number of correct arguments.
            - argument_precision (float): The precision of the arguments.
            - argument_recall (float): The recall of the arguments.
            - argument_f1 (float): The f1 score of the arguments.
    """
    total_event = 0
    correct_event = 0
    total_argument = 0
    predict_argument = 0
    correct_argument = 0

    if not keep_whitespace:
        golds = remove_event_argument_whitespace(golds)
        outputs = remove_event_argument_whitespace(outputs)

    for key, output in outputs.items():
        if real_gold and key not in golds:
            err_msg = f"Key {key} not found in gold."
            raise ValueError(err_msg)

        if not real_gold and key not in golds:
            continue  # This checks that the key is in the gold data
        if not real_gold and "JSON_ERROR" in golds[key]:
            continue  # This checks that the gold data is a valid JSON

        total_event += 1
        predict_argument += len([val for val in output.values() if val])
        total_argument += len([val for val in golds[key].values() if val])

        this_correct_event, this_correct_argument = (
            compare_argument_detection(golds[key], output)
            if not semi
            else semi_compare_argument_detection(golds[key], output)
        )
        correct_event += this_correct_event
        correct_argument += this_correct_argument

    event_accuracy = safe_divide(correct_event, total_event)
    argument_precision = safe_divide(correct_argument, predict_argument)
    argument_recall = safe_divide(correct_argument, total_argument)
    argument_f1 = safe_divide(
        2 * argument_precision * argument_recall, argument_precision + argument_recall
    )

    return {
        "total_event": total_event,
        "correct_event": correct_event,
        "event_accuracy": event_accuracy,
        "total_argument": total_argument,
        "predict_argument": predict_argument,
        "correct_argument": correct_argument,
        "argument_precision": argument_precision,
        "argument_recall": argument_recall,
        "argument_f1": argument_f1,
    }
