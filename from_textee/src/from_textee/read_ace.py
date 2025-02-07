"""This is taken from the TextEE libary and is used to convert the ACE 2005 dataset
into a structed JSON format. It is largely the same as the original code. Some attempt
has been made to document it.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any

import tqdm
from bs4 import BeautifulSoup


def mask_escape(text: str) -> str:
    """Replaces escaped characters with rare sequences.

    Args:
        text (str): text to mask.

    Returns:
        str: masked string.
    """
    return (
        text.replace("&amp;", "ҪҪҪҪҪ").replace("&lt;", "ҚҚҚҚ").replace("&gt;", "ҺҺҺҺ")
    )


def unmask_escape(text: str) -> str:
    """Replaces masking sequences with the original escaped characters.

    Args:
        text (str): masked string.

    Returns:
        str: unmasked string.
    """
    return (
        text.replace("ҪҪҪҪҪ", "&amp;").replace("ҚҚҚҚ", "&lt;").replace("ҺҺҺҺ", "&gt;")
    )


def recover_escape(text: str) -> str:
    """Converts named character references in the given string to the corresponding
    Unicode characters. I didn't notice any numeric character references in this
    dataset.

    Args:
        text (str): text to unescape.

    Returns:
        str: unescaped string.
    """
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self: Span) -> None:
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace("\n", " ")

    def char_offsets_to_token_offsets(
        self: Span, tokens: list[tuple[int, int, str]]
    ) -> None:
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            tokens (List[int, int, str]): a list of token tuples. Each item in
                the list is a triple (start_offset, end_offset, text).
        """
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            err_msg = f"Failed to update offsets for \
                {self.start}-{self.end}:{self.text} in {tokens}"
            raise ValueError(err_msg)
        self.start, self.end = start_, end_

    def to_dict(self: Span) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            dict: a dict of instance variables.
        """
        return {
            "text": recover_escape(self.text),
            "start": self.start,
            "end": self.end,
        }

    def remove_space(self: Span) -> None:
        """Removes heading and trailing spaces in the span text."""
        # heading spaces
        text = self.text.lstrip()
        self.start += len(self.text) - len(text)
        # trailing spaces
        after_text = text.rstrip()
        # self.end = self.start + len(text)
        # self.text = text
        self.end += len(after_text) - len(text)
        self.text = after_text


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    entity_subtype: str
    mention_type: str
    value: str | None = None

    def to_dict(self: Entity) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            "text": recover_escape(self.text),
            # 'entity_id': self.entity_id,
            "id": self.mention_id,  # mention id as id
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "entity_subtype": self.entity_subtype,
            "mention_type": self.mention_type,
        }
        if self.value:
            entity_dict["value"] = self.value
        return entity_dict


@dataclass
class RelationArgument:
    mention_id: str
    role: str
    text: str
    pointed_entity: Entity

    def to_dict(self: RelationArgument) -> dict[str, str | int]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "entity_id": self.mention_id,
            "role": self.role,
            "text": recover_escape(self.text),
            "start": self.pointed_entity.start,
            "end": self.pointed_entity.end,
        }


@dataclass
class RelationArgumentWithoutEntity:
    mention_id: str
    role: str
    text: str


@dataclass
class Relation:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self: Relation) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "id": self.relation_id,
            "relation_type": self.relation_type,
            "relation_subtype": self.relation_subtype,
            "arguments": [self.arg1.to_dict(), self.arg2.to_dict()],
        }


@dataclass
class RelationWithoutEntity:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgumentWithoutEntity
    arg2: RelationArgumentWithoutEntity


@dataclass
class EventArgument:
    mention_id: str
    role: str
    text: str
    pointed_entity: Entity

    def to_dict(self: EventArgument) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "entity_id": self.mention_id,
            "role": self.role,
            "text": recover_escape(self.text),
            "start": self.pointed_entity.start,
            "end": self.pointed_entity.end,
        }


@dataclass
class EventArgumentWthoutEnity:
    mention_id: str
    role: str
    text: str


@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: list[EventArgument]

    def to_dict(self: Event) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            # 'evnet_id': self.event_id,
            "id": self.mention_id,
            "event_main_type": self.event_type,
            "event_subtype": self.event_subtype,
            "event_type": f"{self.event_type}:{self.event_subtype}",
            "trigger": self.trigger.to_dict(),
            "arguments": [arg.to_dict() for arg in self.arguments],
        }


@dataclass
class EventWithoutEntity:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: list[EventArgumentWthoutEnity]


@dataclass
class Sentence(Span):
    doc_id: str
    sent_id: str
    tokens: list[str]
    entities: list[Entity]
    relations: list[Relation]
    events: list[Event]
    doc_tokens: list[str]
    lang: str = "en"

    def to_dict(self: Sentence) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "doc_id": self.doc_id,
            "wnd_id": self.sent_id,
            "text": recover_escape(self.text).replace("\t", " "),
            "lang": self.lang,
            "tokens": [recover_escape(t) for t in self.tokens],
            "entity_mentions": [entity.to_dict() for entity in self.entities],
            "event_mentions": [event.to_dict() for event in self.events],
            "relation_mentions": [relation.to_dict() for relation in self.relations],
        }


@dataclass
class Document:
    doc_id: str
    sentences: list[Sentence]

    def to_dict(self: Document) -> dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "doc_id": self.doc_id,
            "sentences": [sent.to_dict() for sent in self.sentences],
        }


def tokenize(
    sentence: tuple[str, int, int],
    doc_id: str,
    token_map: dict[str, dict[str, list[list[int]]]],
) -> list[tuple[int, int, str]]:
    text, start, end = sentence
    return [
        (t[0], t[1], text[t[0] - start : t[1] - start])
        for t in token_map[doc_id][f"{start},{end}"]
    ]


def process_entities(
    entities: list[Entity], sentences: list[tuple[str, int, int]]
) -> tuple[list[list[Entity]], int]:
    """
    Cleans entities and splits them into lists
    """
    sentence_entities: list[list[Entity]] = [[] for _ in range(len(sentences))]
    cross_boundary_entity_count = 0

    # assign each entity to the sentence where it appears
    for entity in entities:
        start, end = entity.start, entity.end
        flag = False
        for i, (sent_text, s, e) in enumerate(sentences):
            if start >= s and end <= e:
                sentence_entities[i].append(entity)
                flag = True
                # check label and text consistency
                assert entity.text == recover_escape(
                    sent_text[start - s : end - s]
                )  # , ipdb.set_trace()
                break

            # TODO: Assume no entities that crosses sentence boundary, which could be
            # wrong.
        if not flag:
            cross_boundary_entity_count += 1

    return sentence_entities, cross_boundary_entity_count


def process_relation(
    relations: list[RelationWithoutEntity],
    sentence_entities: list[list[Entity]],
    sentences: list[tuple[str, int, int]],
) -> tuple[list[list[Relation]], int]:
    sentence_relations: list[list[Relation]] = [[] for _ in range(len(sentences))]
    cross_sentence_relation_count = 0
    for relation in relations:
        mention_id1 = relation.arg1.mention_id
        mention_id2 = relation.arg2.mention_id
        flag = False
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = any(mention_id1 == e.mention_id for e in entities)
            arg2_in_sent = any(mention_id2 == e.mention_id for e in entities)
            if arg1_in_sent and arg2_in_sent:
                # Ensure the entities are in the same sentence

                # Update relation argument
                entmap = {e.mention_id: e for e in entities}
                sentence_relations[i].append(
                    Relation(
                        relation.relation_id,
                        relation.relation_type,
                        relation.relation_subtype,
                        RelationArgument(
                            relation.arg1.mention_id,
                            relation.arg1.role,
                            entmap[relation.arg1.mention_id].text,
                            entmap[relation.arg1.mention_id],
                        ),
                        RelationArgument(
                            relation.arg2.mention_id,
                            relation.arg2.role,
                            entmap[relation.arg2.mention_id].text,
                            entmap[relation.arg2.mention_id],
                        ),
                    )
                )
                flag = True
                break
        if not flag:
            cross_sentence_relation_count += 1
    return sentence_relations, cross_sentence_relation_count


def process_events(
    events: list[EventWithoutEntity],
    sentence_entities: list[list[Entity]],
    sentences: list[tuple[str, int, int]],
) -> tuple[list[list[Event]], int, int]:
    sentence_events: list[list[Event]] = [[] for _ in range(len(sentences))]
    # assign each event mention to the sentence where it appears
    cross_boundary_trigger_count = 0
    cross_boundary_argument_count = 0
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        flag_trigger = False
        for i, (sent_text, s, e) in enumerate(sentences):
            # check trigger
            # TODO: Here, we assume no cross single sentence boundary trigger
            if start >= s and end <= e:
                flag_trigger = True

                # check label and text consistency
                assert event.trigger.text == recover_escape(
                    sent_text[start - s : end - s]
                )  # , ipdb.set_trace()

                sent_entities = sentence_entities[i]
                # clean the argument list
                arguments = []
                for argument in event.arguments:
                    flag_argument = False
                    mention_id = argument.mention_id
                    for entity in sent_entities:
                        if entity.mention_id == mention_id:
                            arguments.append(
                                EventArgument(
                                    argument.mention_id,
                                    argument.role,
                                    entity.text,
                                    entity,
                                )
                            )
                            flag_argument = True
                            break
                    if not flag_argument and ("-E" in argument.mention_id):
                        cross_boundary_argument_count += 1
                event_cleaned = Event(
                    event.event_id,
                    event.mention_id,
                    event.event_type,
                    event.event_subtype,
                    trigger=event.trigger,
                    arguments=arguments,
                )
                sentence_events[i].append(event_cleaned)
                break  # no need to iterate sentences
        if not flag_trigger:
            cross_boundary_trigger_count += 1
    return sentence_events, cross_boundary_trigger_count, cross_boundary_argument_count


TAG_PATTERN = re.compile("<[^<>]+>", re.MULTILINE)


def read_sgm_file(
    path: str, sent_map: dict[str, dict[str, list[list[int]]]]
) -> list[tuple[str, int, int]]:
    with open(path, encoding="utf-8") as fp:
        data = fp.read()

    # Chunk the document
    chunks = TAG_PATTERN.sub("⁑", data).split("⁑")

    # Get the offset of <TEXT>
    data_update = data.replace("<TEXT>", "⁂")
    data_update = TAG_PATTERN.sub("", data_update)
    min_offset = max(0, data_update.find("⁂"))
    data_update = data_update.replace("⁂", "")

    # Extract sentences from chunks
    chunk_offset = 0
    sentences = []
    for chunk in chunks:
        lines = chunk.split("\n")
        current_sentence = []
        start = offset = 0
        for line in lines:
            offset += len(line) + 1
            if line.strip():
                current_sentence.append(line)
            else:
                # empty line
                if current_sentence:
                    sentence = " ".join(
                        current_sentence
                    )  # TODO: This could be problematic for Chinese, we cannot
                    # directly use ''.join to combine, since the offset will have
                    # issues.
                    if start + chunk_offset >= min_offset:
                        sentences.append(
                            (
                                sentence,
                                start + chunk_offset,
                                start + chunk_offset + len(sentence),
                            )
                        )
                    current_sentence = []
                start = offset
        if current_sentence:
            sentence = " ".join(current_sentence)
            if start + chunk_offset >= min_offset:
                sentences.append(
                    (
                        sentence,
                        start + chunk_offset,
                        start + chunk_offset + len(sentence),
                    )
                )
        chunk_offset += len(chunk)

    # Re-tokenize sentences
    doc_id = path.rsplit("/", 1)[1][:-4]
    sentences_ = [
        [
            (sent[0][s[0] - sent[1] : s[1] - sent[1]], s[0], s[1])
            for s in sent_map[doc_id][f"{sent[1]},{sent[2]}"]
        ]
        for sent in sentences
    ]
    return [s for sent in sentences_ for s in sent]


def read_apf_file(
    path: str, time_and_val: bool, use_full_span: bool
) -> tuple[
    str, str, list[Entity], list[RelationWithoutEntity], list[EventWithoutEntity]
]:
    with open(path, encoding="utf-8") as fp:
        soup = BeautifulSoup(fp.read(), "lxml-xml")

    # metadata
    root = soup.find("source_file")
    source = root["SOURCE"]
    doc = root.find("document")
    doc_id = doc["DOCID"]

    entity_list, relation_list, event_list = [], [], []

    # entities: nam, nom, pro
    for entity in doc.find_all("entity"):
        entity_id = entity["ID"]
        entity_type = entity["TYPE"]
        entity_subtype = entity["SUBTYPE"]
        for entity_mention in entity.find_all("entity_mention"):
            mention_id = entity_mention["ID"]
            mention_type = entity_mention["TYPE"]
            if use_full_span:
                fullspan = entity_mention.find("extent").find("charseq")
                start, end, text = (
                    int(fullspan["START"]),
                    int(fullspan["END"]),
                    fullspan.text,
                )
            else:
                head = entity_mention.find("head").find("charseq")
                start, end, text = int(head["START"]), int(head["END"]), head.text
            entity_list.append(
                Entity(
                    start,
                    end + 1,
                    text,
                    entity_id,
                    mention_id,
                    entity_type,
                    entity_subtype,
                    mention_type,
                )
            )

    if time_and_val:
        # entities: value
        for entity in doc.find_all("value"):
            entity_id = entity["ID"]
            entity_type = entity["TYPE"]
            entity_subtype = entity.get("SUBTYPE", None)
            for entity_mention in entity.find_all("value_mention"):
                mention_id = entity_mention["ID"]
                mention_type = "VALUE"
                extent = entity_mention.find("extent").find("charseq")
                start, end, text = int(extent["START"]), int(extent["END"]), extent.text
                entity_list.append(
                    Entity(
                        start,
                        end + 1,
                        text,
                        entity_id,
                        mention_id,
                        entity_type,
                        entity_subtype,
                        mention_type,
                    )
                )

        # entities: timex
        for entity in doc.find_all("timex2"):
            entity_id = entity["ID"]
            entity_type = entity_subtype = "TIME"
            value = entity.get("VAL", None)
            for entity_mention in entity.find_all("timex2_mention"):
                mention_id = entity_mention["ID"]
                mention_type = "TIME"
                extent = entity_mention.find("extent").find("charseq")
                start, end, text = int(extent["START"]), int(extent["END"]), extent.text
                entity_list.append(
                    Entity(
                        start,
                        end + 1,
                        text,
                        entity_id,
                        mention_id,
                        entity_type,
                        entity_subtype,
                        mention_type,
                        value=value,
                    )
                )

    # relations
    for relation in doc.find_all("relation"):
        relation_type = relation["TYPE"]
        if relation_type == "METONYMY":
            continue
        relation_subtype = relation["SUBTYPE"]
        for relation_mention in relation.find_all("relation_mention"):
            mention_id = relation_mention["ID"]
            arg1 = arg2 = None
            for arg in relation_mention.find_all("relation_mention_argument"):
                arg_mention_id = arg["REFID"]
                arg_role = arg["ROLE"]
                arg_text = arg.find("extent").find("charseq").text
                if arg_role == "Arg-1":
                    arg1 = RelationArgumentWithoutEntity(
                        arg_mention_id, arg_role, arg_text
                    )
                elif arg_role == "Arg-2":
                    arg2 = RelationArgumentWithoutEntity(
                        arg_mention_id, arg_role, arg_text
                    )
            if arg1 and arg2:
                relation_list.append(
                    RelationWithoutEntity(
                        mention_id, relation_type, relation_subtype, arg1, arg2
                    )
                )

    # events
    for event in doc.find_all("event"):
        event_id = event["ID"]
        event_type = event["TYPE"]
        event_subtype = event["SUBTYPE"]
        for event_mention in event.find_all("event_mention"):
            mention_id = event_mention["ID"]
            trigger = event_mention.find("anchor").find("charseq")
            trigger_start, trigger_end = int(trigger["START"]), int(trigger["END"])
            trigger_text = trigger.text
            event_args = []
            for arg in event_mention.find_all("event_mention_argument"):
                arg_mention_id = arg["REFID"]
                arg_role = arg["ROLE"]
                arg_text = arg.find("extent").find("charseq").text
                event_args.append(
                    EventArgumentWthoutEnity(arg_mention_id, arg_role, arg_text)
                )
            event_list.append(
                EventWithoutEntity(
                    event_id,
                    mention_id,
                    event_type,
                    event_subtype,
                    Span(trigger_start, trigger_end + 1, trigger_text),
                    event_args,
                )
            )

    # remove heading/tailing spaces
    for entity in entity_list:
        entity.remove_space()
    for event in event_list:
        event.trigger.remove_space()

    return (doc_id, source, entity_list, relation_list, event_list)


def convert(
    sgm_file: str,
    apf_file: str,
    sent_map: dict[str, dict[str, list[list[int]]]],
    token_map: dict[str, dict[str, list[list[int]]]],
    time_and_val: bool,
    use_full_span: bool,
) -> tuple[Document, int, int, int, int]:
    sentences = read_sgm_file(sgm_file, sent_map)
    # sentences is a list of tuple (sentence, start_char_index, end_char_index)

    doc_id, _source, entities, relations, events = read_apf_file(
        apf_file, time_and_val=time_and_val, use_full_span=use_full_span
    )

    # Process entities, relations, and events
    sentence_entities, cross_boundary_entity_count = process_entities(
        entities, sentences
    )
    sentence_relations, cross_boundary_relation_count = process_relation(
        relations, sentence_entities, sentences
    )
    sentence_events, cross_boundary_trigger_count, cross_boundary_argument_count = (
        process_events(events, sentence_entities, sentences)
    )

    #####
    # Right now, all the annotation is under characater span level.
    #####

    # Tokenization
    sentence_tokens = [tokenize(s, doc_id, token_map) for s in sentences]

    # Convert span character offsets to token indices
    sentence_objs = []
    doc_tokens = [tok for sent in sentence_tokens for tok in sent]
    for i, (toks, ents, evts, rels, sent) in enumerate(
        zip(
            sentence_tokens,
            sentence_entities,
            sentence_events,
            sentence_relations,
            sentences,
            strict=False,
        )
    ):
        for entity in ents:
            entity.char_offsets_to_token_offsets(toks)
        for event in evts:
            event.trigger.char_offsets_to_token_offsets(toks)
        wnd_id = f"{doc_id}-{i}"
        sent_obj = Sentence(
            start=sent[1],
            end=sent[2],
            text=sent[0],
            doc_id=doc_id,
            sent_id=wnd_id,
            tokens=[t for _, _, t in toks],
            entities=ents,
            relations=rels,
            events=evts,
            doc_tokens=[tok for sent in sentence_tokens for _, _, tok in sent],
        )
        sent_obj.remove_space()
        # try:
        sent_obj.char_offsets_to_token_offsets(doc_tokens)
        # except Exception:
        # ipdb.set_trace()
        sentence_objs.append(sent_obj)
    return (
        Document(doc_id, sentence_objs),
        cross_boundary_entity_count,
        cross_boundary_relation_count,
        cross_boundary_trigger_count,
        cross_boundary_argument_count,
    )


def convert_batch(
    input_path: str | PosixPath,
    sent_map: dict[str, dict[str, list[list[int]]]],
    token_map: dict[str, dict[str, list[list[int]]]],
    time_and_val: bool = False,
    use_full_span: bool = False,
) -> list[dict]:
    # Read all file
    sgm_files = glob.glob(os.path.join(str(input_path), "**", "timex2norm", "*.sgm"))
    print(input_path)
    print("Converting the dataset to JSON format")
    print(f"#SGM files: {len(sgm_files)}")
    progress = tqdm.tqdm(total=len(sgm_files))
    (
        cross_boundary_entity_counts,
        cross_boundary_relation_counts,
        cross_boundary_trigger_counts,
        cross_boundary_argument_counts,
    ) = (
        0,
        0,
        0,
        0,
    )
    objs = []
    for sgm_file in sgm_files:
        progress.update(1)
        apf_file = sgm_file.replace(".sgm", ".apf.xml")
        (
            doc,
            cross_boundary_entity_count,
            cross_boundary_relation_count,
            cross_boundary_trigger_count,
            cross_boundary_argument_count,
        ) = convert(
            sgm_file,
            apf_file,
            sent_map,
            token_map,
            time_and_val=time_and_val,
            use_full_span=use_full_span,
        )
        cross_boundary_argument_counts += cross_boundary_argument_count
        cross_boundary_entity_counts += cross_boundary_entity_count
        cross_boundary_trigger_counts += cross_boundary_trigger_count
        cross_boundary_relation_counts += cross_boundary_relation_count

        # This use to be in a bool that was always True
        objs.append(doc.to_dict())

    print(f"Cross Boundary Entity: {cross_boundary_entity_counts}")
    print(f"Cross Boundary Relation: {cross_boundary_relation_counts}")
    print(f"Cross Boundary Trigger: {cross_boundary_trigger_counts}")
    print(f"Cross Boundary Argument: {cross_boundary_argument_counts}")
    progress.close()

    return objs
