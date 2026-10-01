import numpy as np

from editing.inpaint import feather_blend
from editing.insert import square_crop_box
from editing.segment import build_removal_mask, mask_iou, placement_mask, polygon_to_mask, removal_mask

SHAPE = (200, 300, 3)


def test_polygon_to_mask_fills_inside():
    mask = polygon_to_mask(np.array([[50, 50], [150, 50], [150, 120], [50, 120]], float), SHAPE)
    assert mask.shape == SHAPE[:2] and mask[80, 100] == 255 and mask[10, 10] == 0
    assert polygon_to_mask(None, SHAPE).sum() == 0


def test_removal_mask_covers_box_dilation_and_shadow():
    obj = np.zeros(SHAPE[:2], np.uint8)
    obj[60:100, 100:160] = 255
    mask = removal_mask(obj, (100, 60, 160, 100), SHAPE, grow=5, shadow_ratio=0.5)
    assert mask[80, 130] == 255  # object
    assert mask[80, 97] == 255  # dilated border
    assert mask[115, 130] == 255  # shadow band below the box (40 px * 0.5 = 20 px)
    assert mask[150, 130] == 0


def test_build_removal_mask_unions_instances():
    a = np.zeros(SHAPE[:2], np.uint8)
    b = np.zeros(SHAPE[:2], np.uint8)
    a[10:30, 10:30] = 255
    b[150:170, 250:270] = 255
    mask = build_removal_mask([a, b], [(10, 10, 30, 30), (250, 150, 270, 170)], SHAPE, grow=1, shadow_ratio=0)
    assert mask[20, 20] == 255 and mask[160, 260] == 255


def test_placement_mask_keeps_bottom_and_adds_headroom():
    mask = placement_mask((100, 50, 200, 150), SHAPE)
    ys, xs = np.where(mask > 0)
    assert ys.min() < 50 and xs.min() < 100 and xs.max() >= 199 and ys.max() <= 153


def test_mask_iou():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[:, :5] = 255
    b[:, :5] = 255
    assert mask_iou(a, b) == 1.0
    b[:, :] = 0
    b[:, 5:] = 255
    assert mask_iou(a, b) == 0.0


def test_square_crop_stays_inside_image():
    x1, y1, x2, y2 = square_crop_box((250, 150, 299, 199), SHAPE)
    assert x2 - x1 == y2 - y1
    assert 0 <= x1 and 0 <= y1 and x2 <= SHAPE[1] and y2 <= SHAPE[0]


def test_feather_blend_keeps_outside_pixels():
    original = np.zeros(SHAPE, np.uint8)
    edited = np.full(SHAPE, 200, np.uint8)
    mask = np.zeros(SHAPE[:2], np.uint8)
    mask[50:150, 100:200] = 255
    out = feather_blend(original, edited, mask, feather=3)
    assert out[100, 150, 0] > 190 and out[5, 5, 0] == 0
