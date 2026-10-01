from editing.ground import select_detections
from editing.types import Detection

SHAPE = (500, 500, 3)
FAR_LEFT_SMALL = Detection((10, 100, 40, 160), 0.90)  # distant pedestrian
LEFT = Detection((150, 50, 250, 400), 0.80)
RIGHT = Detection((300, 50, 400, 400), 0.95)


def test_best_is_most_confident():
    assert select_detections([LEFT, RIGHT], "best", SHAPE) == [RIGHT]


def test_positional_choice_ignores_small_background_instances():
    dets = [FAR_LEFT_SMALL, LEFT, RIGHT]
    assert select_detections(dets, "left", SHAPE) == [LEFT]
    assert select_detections(dets, "right", SHAPE) == [RIGHT]


def test_size_and_all():
    dets = [FAR_LEFT_SMALL, LEFT, RIGHT]
    assert select_detections(dets, "smallest", SHAPE) == [FAR_LEFT_SMALL]
    assert select_detections(dets, "largest", SHAPE)[0] in (LEFT, RIGHT)
    assert select_detections(dets, "all", SHAPE) == dets


def test_unknown_which_falls_back_to_best_and_empty_input():
    assert select_detections([LEFT, RIGHT], "somewhere", SHAPE) == [RIGHT]
    assert select_detections([], "left", SHAPE) == []
