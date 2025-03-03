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


def stats_on_event_detection(
    golds: dict[str, list], outputs: dict[str, list]
) -> dict[str, int | float]:
    """
    Calculate the statistics on the event detection task.

    Args:
        golds (dict[str, dict | list]): The gold data in the format of a dictionary with
            the key as the key and the 'output' as the value.
        outputs (dict[str, dict | list]): The output data in the format of a dictionary
            with the key as the key and the 'output' as the value.

    Returns:
        dict[str, int | float]: This is a dict with the following keys:
            - total_passage (int): The total number of passages.
            - passage_accuracy (float): The accuracy of the passage level.
            - correct_passage (int): The total number of correct passages.
            - total_event (int): The total number of events in the gold data.
            - predict_event (int): The total number of events in the output data.
            - correct_all (int): The total number of correct events in the output data.
            - event_precision (float): The percision of the events.
            - event_recall (float): The recall of the events.
            - event_f1 (float): The f1 score of the events.
            - event_type_precision (float): The percision of the event types.
            - event_type_recall (float): The recall of the event types.
            - event_type_f1 (float): The f1 score of the event types.
    """

    total_passage = 0
    correct_passage = 0
    total_event = 0
    predict_event = 0
    correct_type = 0
    correct_all = 0

    for key, output in outputs.items():
        if key not in golds:
            err_msg = f"Key {key} not found in gold."
            raise ValueError(err_msg)

        total_passage += 1
        if output == golds[key]:
            correct_passage += 1

        total_event += len(golds[key])

        if not isinstance(output, list):
            # Need the type checks as llm can misbehave
            continue  # type: ignore[unreachable]

        predict_event += len(output)
        if len(output) == 0 or len(golds[key]) == 0:
            continue

        for g_event in golds[key]:
            event_correct = False
            event_type_correct = False
            for o_event in output:
                if g_event == o_event:
                    event_correct = True
                if g_event.keys() == o_event.keys():
                    event_type_correct = True

            if event_correct:
                correct_all += 1
            if event_type_correct:
                correct_type += 1

    passage_accuracy = correct_passage / total_passage
    event_precision = correct_all / predict_event
    event_recall = correct_all / total_event
    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall)
    event_type_precision = correct_type / predict_event
    event_type_recall = correct_type / total_event
    event_type_f1 = (
        2
        * event_type_precision
        * event_type_recall
        / (event_type_precision + event_type_recall)
    )

    return {
        "total_passage": total_passage,
        "correct_passage": correct_passage,
        "passage_accuracy": passage_accuracy,
        "total_event": total_event,
        "predict_event": predict_event,
        "correct_all": correct_all,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "event_type_precision": event_type_precision,
        "event_type_recall": event_type_recall,
        "event_type_f1": event_type_f1,
    }


def stats_on_argument_detection(
    golds: dict[str, dict], outputs: dict[str, dict]
) -> dict[str, int | float]:
    """
    Calculate the statistics on the argument detection task.

    Args:
        golds (dict[str, dict | list]): The gold data in the format of a dictionary with
            the key as the key and the 'output' as the value.
        outputs (dict[str, dict | list]): The output data in the format of a dictionary
            with the key as the key and the 'output' as the value.

    Returns:
        dict[str, int | float]: This is a dict with the following values:
            - total_event (int): The total number of events in the gold data.
            - correct_event (int): The total number of correct events in the data.
            - event_accuracy (float): The accuracy of the events.
            - total_argument (int): The total number of arguments in the gold data.
            - predict_argument (int): The total number of arguments in the output data.
            - correct_argument (int): The total number of correct arguments.
            - argument_precision (float): The percision of the arguments.
            - argument_recall (float): The recall of the arguments.
            - argument_f1 (float): The f1 score of the arguments.
    """
    total_event = 0
    correct_event = 0
    total_argument = 0
    predict_argument = 0
    correct_argument = 0

    for key, output in outputs.items():
        if key not in golds:
            err_msg = f"Key {key} not found in gold."
            raise ValueError(err_msg)

        total_event += 1
        correct_event += 1 if output == golds[key] else 0

        predict_argument += len([val for val in output.values() if val])
        total_argument += len([val for val in golds[key].values() if val])
        for arg_type, arg in golds[key].items():
            if arg and arg_type in output:
                correct_argument += 1 if output[arg_type] == arg else 0

    event_accuracy = correct_event / total_event
    argument_precision = correct_argument / predict_argument
    argument_recall = correct_argument / total_argument
    argument_f1 = (
        2
        * argument_precision
        * argument_recall
        / (argument_precision + argument_recall)
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
