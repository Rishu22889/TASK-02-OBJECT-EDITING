import json
from pathlib import Path

import pytest

from editing.config import ParserConfig
from editing.parse import CascadeParser, FallbackParser, LLMParser, RegexParser, extract_json, make_parsed, unusual

TESTSET = json.loads((Path(__file__).resolve().parents[1] / "data" / "parser_testset.json").read_text())


def test_regex_remove_and_replace():
    regex = RegexParser()
    p = regex("Remove the car")
    assert (p.action, p.object, p.coco_class, p.which) == ("remove", "car", "car", "best")
    p = regex("swap the SUV for a red bicycle")
    assert (p.action, p.coco_class, p.target) == ("replace", "car", "red bicycle")
    assert "red bicycle" in p.target_prompt


@pytest.mark.parametrize(
    "instruction, action, obj, which, target",
    [
        ("Can you get rid of the guy on the left?", "remove", "guy", "left", ""),
        ("I don't want that sedan in my photo anymore", "remove", "sedan", "best", ""),
        ("make the kitten disappear", "remove", "kitten", "best", ""),
        ("remove all of the people from the background", "remove", "people", "all", ""),
        ("delete the umbrellas", "remove", "umbrellas", "all", ""),
        ("take out the biggest car", "remove", "car", "largest", ""),
        ("remove the man holding all the balloons", "remove", "man", "best", ""),
        ("remove the hanging lamp", "remove", "hanging lamp", "best", ""),
        ("get rid of the candle, it looks ugly", "remove", "candle", "best", ""),
        ("the bike should be a motorcycle instead", "replace", "bike", "best", "motorcycle"),
        ("i want the cat replaced by a rabbit", "replace", "cat", "best", "rabbit"),
        ("put a lamp instead of the vase", "replace", "vase", "best", "lamp"),
        ("turn the horse on the right into a zebra", "replace", "horse", "right", "zebra"),
    ],
)
def test_regex_phrasing_and_instance_words(instruction, action, obj, which, target):
    p = RegexParser()(instruction)
    assert (p.action, p.object, p.which, p.target) == (action, obj, which, target)
    assert not unusual(instruction, p)


@pytest.mark.parametrize(
    "instruction",
    [
        "remove the thing on the left",  # vague object
        "remove the red and white striped umbrella",  # long object phrase
        "replace the man with the hat with a robot",  # two separators
        "replace the dog with a big fluffy white persian cat sitting",  # long target
        "don't remove the dog, remove the cat",  # negation
    ],
)
def test_unusual_instructions_are_flagged(instruction):
    assert unusual(instruction, RegexParser()(instruction))


def test_regex_cannot_parse_free_form():
    with pytest.raises(ValueError):
        RegexParser()("the cars should be bikes")


def test_regex_score_on_testset():
    """26 of 26 fully right (the E4 baseline rules got 14, 54 %).

    The rules were written with this set in view, so it is a regression check, not a held-out score.
    """
    regex = RegexParser()
    correct = 0
    for case in TESTSET:
        try:
            p = regex(case["instruction"])
        except ValueError:
            continue
        correct += (
            p.action == case["action"]
            and p.coco_class in case["coco"]
            and any(k in p.object for k in case["object_keywords"])
            and p.which == case["which"]
            and (case["target"] is None or case["target"] in p.target)
        )
    assert len(TESTSET) == 26
    assert correct == 26


def _bare_llm_parser(mapping: str) -> LLMParser:
    parser = object.__new__(LLMParser)  # skip model loading
    parser.cfg = ParserConfig(class_mapping=mapping)
    return parser


def test_alias_mapping_fixes_enum_drift():
    """A 0.5B model forced into the class enum wrote kitten -> kite; alias mapping uses the free text."""
    data = {"action": "remove", "object": "kitten", "coco_class": "kite", "which": "best",
            "target": "", "target_prompt": ""}
    assert _bare_llm_parser("llm").to_parsed(data).coco_class == "kite"
    assert _bare_llm_parser("alias").to_parsed(data).coco_class == "cat"


def test_llm_output_is_normalised():
    data = {"action": "replace", "object": "The SUV", "coco_class": "sports ball", "which": "nowhere",
            "target": "a red bicycle", "target_prompt": ""}
    p = _bare_llm_parser("alias").to_parsed(data)
    assert (p.object, p.coco_class, p.which, p.target) == ("suv", "car", "best", "red bicycle")
    assert p.target_prompt  # filled with the default template


def test_extract_json_tolerates_chatter():
    assert extract_json('Sure! {"action": "remove"} done')["action"] == "remove"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_fallback_parser_uses_regex_on_failure():
    def broken(_):
        raise RuntimeError("model crashed")

    p = FallbackParser(broken)("remove the dog")
    assert p.parser == "regex" and p.coco_class == "dog"


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, instruction):
        self.calls.append(instruction)
        return make_parsed("remove", "dog", parser="llm")


def test_cascade_uses_regex_without_loading_the_llm():
    loads = []
    cascade = CascadeParser(lambda: loads.append(1) or _FakeLLM())
    p = cascade("remove the car on the left")
    assert (p.parser, p.which) == ("regex", "left") and not loads


def test_cascade_sends_unusual_instructions_to_the_llm_once_loaded():
    llm, loads = _FakeLLM(), []
    cascade = CascadeParser(lambda: loads.append(1) or llm)
    assert cascade("remvoe the dog").parser == "llm"  # no pattern
    assert cascade("remove the thing on the left").parser == "llm"  # vague object
    assert len(llm.calls) == 2 and len(loads) == 1


def test_cascade_keeps_the_regex_parse_when_the_llm_is_unavailable():
    loads = []

    def missing():
        loads.append(1)
        raise ImportError("llama-cpp-python not installed")

    cascade = CascadeParser(missing)
    p = cascade("remove the red and white striped umbrella")
    assert p.parser == "regex" and p.object == "red and white striped umbrella"
    with pytest.raises(ValueError):
        cascade("remvoe the dog")
    assert len(loads) == 1  # a failed load is not retried


def test_make_parsed_drops_target_for_remove():
    p = make_parsed("remove", "the cat", target="dog")
    assert p.target == "" and p.target_prompt == ""
