"""Object masks, the removal mask and the placement mask for new objects."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from editing.utils import file_mb
from editing.weights import ensure_weights


class MobileSamSegmenter:
    """MobileSAM (TinyViT encoder): box prompt -> mask. Used on the open-vocabulary path."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        from mobile_sam import SamPredictor, sam_model_registry

        path = ensure_weights(Path(checkpoint))
        model = sam_model_registry["vit_t"](checkpoint=str(path))
        model.to(device).eval()
        self.predictor = SamPredictor(model)
        self.size_mb = file_mb(path)

    def segment(self, image: np.ndarray, boxes: list) -> list[np.ndarray]:
        """Encode the image once, then one box prompt per instance. Returns uint8 masks (255 = object)."""
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks = []
        for box in boxes:
            candidates, scores, _ = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=True
            )
            masks.append(candidates[int(np.argmax(scores))].astype(np.uint8) * 255)
        return masks


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mask) > 127, 255, 0).astype(np.uint8)


def polygon_to_mask(polygon: np.ndarray | None, image_shape: tuple) -> np.ndarray:
    mask = np.zeros(image_shape[:2], np.uint8)
    if polygon is not None and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


def removal_mask(
    mask: np.ndarray, bbox: tuple, image_shape: tuple, grow: int = 15, shadow_ratio: float = 0.15
) -> np.ndarray:
    """Dilated object mask OR its box, extended downward over the contact shadow (255 = remove).

    Box-like and generous on purpose: inpainters smear any object pixel or shadow
    left at the border, and an exact silhouette is not needed once it is dilated.
    """
    h, w = image_shape[:2]
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=grow)
    x1, y1, x2, y2 = map(int, bbox)
    y2 = y2 + int((y2 - y1) * shadow_ratio)
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    box = np.zeros_like(dilated)
    if x2 > x1 and y2 > y1:
        box[y1:y2, x1:x2] = 255
    return cv2.bitwise_or(dilated, box)


def build_removal_mask(
    masks: list[np.ndarray], boxes: list, image_shape: tuple, grow: int = 15, shadow_ratio: float = 0.15
) -> np.ndarray:
    out = np.zeros(image_shape[:2], np.uint8)
    for mask, box in zip(masks, boxes):
        out = cv2.bitwise_or(out, removal_mask(mask, box, image_shape, grow, shadow_ratio))
    return out


def placement_mask(bbox: tuple, image_shape: tuple, grow: float = 0.08, headroom: float = 0.15) -> np.ndarray:
    """Where a new object may appear: the old box, a little wider and taller, bottom edge kept.

    Box-shaped rather than the old silhouette, which would force the old object's shape.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, x1 - int(bw * grow)), min(w, x2 + int(bw * grow))
    y1 = max(0, y1 - int(bh * headroom))
    y2 = min(h, y2 + int(bh * 0.03))
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 127, b > 127
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
