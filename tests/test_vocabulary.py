import pytest

from editing.vocabulary import COCO_NAMES, clean_phrase, to_coco_class


def test_coco_list_matches_ultralytics_order():
    assert len(COCO_NAMES) == 80
    assert COCO_NAMES[0] == "person" and COCO_NAMES[2] == "car" and COCO_NAMES[-1] == "toothbrush"


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("car", "car"),
        ("the dark sedan on the highway", "car"),
        ("guy on the left", "person"),
        ("kitten", "cat"),
        ("suv", "car"),
        ("cyclist", "person"),
        ("people from the background", "person"),
        ("man walking near the car", "person"),  # first noun wins
        ("cell phone", "cell phone"),
        ("phone", "cell phone"),
        ("cars", "car"),
        ("scissors", "scissors"),
        ("orange car", "car"),  # colour word before a noun is a modifier
        ("the orange", "orange"),
        ("teh car", "car"),
        ("hat", "none"),
        ("wooden ladder", "none"),
        ("", "none"),
    ],
)
def test_to_coco_class(phrase, expected):
    assert to_coco_class(phrase) == expected


def test_clean_phrase_strips_articles_and_politeness():
    assert clean_phrase("The red car please.") == "red car"
    assert clean_phrase("all the people") == "people"
    assert clean_phrase("an apple!") == "apple"
