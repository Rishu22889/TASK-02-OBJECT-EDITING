"""Grounding: find the instance(s) the instruction refers to."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from editing.types import Detection
from editing.utils import file_mb
from editing.weights import ensure_weights, find_clip_text_encoders


def detections_from_result(result, class_id: int | None = None) -> list[Detection]:
    """Convert one Ultralytics `Results` object into Detections (with polygons if it has masks)."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    polygons = result.masks.xy if getattr(result, "masks", None) is not None else None
    detections = []
    for i, box in enumerate(result.boxes):
        if class_id is not None and int(box.cls.item()) != class_id:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf.item()),
                polygon=polygons[i] if polygons is not None else None,
            )
        )
    return detections


class CocoDetector:
    """YOLOv8 on the 80 COCO classes. With a `-seg` checkpoint the masks come from the same pass."""

    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO

        path = Path(weights)
        if path.suffix == ".pt":
            path = ensure_weights(path)
        task = "segment" if "seg" in path.name else None
        self.model = YOLO(str(path), task=task) if task else YOLO(str(path))
        self.class_ids = {name.lower(): idx for idx, name in self.model.names.items()}
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.size_mb = file_mb(path)

    def detect(self, image: np.ndarray, class_name: str) -> list[Detection]:
        class_id = self.class_ids.get(class_name)
        if class_id is None:
            return []
        result = self.model.predict(
            source=image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        return detections_from_result(result, class_id)


class OpenVocabDetector:
    """YOLO-World: the class list is a CLIP text embedding of the phrase, computed at runtime."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.15,
        min_conf: float = 0.05,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        from ultralytics import YOLOWorld

        path = ensure_weights(Path(weights))
        self.model = YOLOWorld(str(path))
        self.conf, self.min_conf, self.imgsz, self.device = conf, min_conf, imgsz, device
        self._phrase: str | None = None
        self.set_phrase("person")  # downloads the CLIP text encoder on first use
        self.size_mb = file_mb(path) + sum(file_mb(p) for p in find_clip_text_encoders())

    def set_phrase(self, phrase: str) -> float:
        """Encode the phrase as the only class (cached). Returns seconds spent."""
        if phrase == self._phrase:
            return 0.0
        start = time.perf_counter()
        self.model.set_classes([phrase])
        self._phrase = phrase
        return time.perf_counter() - start

    def detect(self, image: np.ndarray, phrase: str) -> list[Detection]:
        """Boxes with confidence >= conf; if none, the single best box >= min_conf."""
        self.set_phrase(phrase)
        result = self.model.predict(
            source=image, conf=self.min_conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        detections = detections_from_result(result)
        strong = [d for d in detections if d.confidence >= self.conf]
        if not strong and detections:
            strong = [max(detections, key=lambda d: d.confidence)]
        return strong


def select_detections(
    detections: list[Detection], which: str, image_shape: tuple, min_rel_area: float = 0.2
) -> list[Detection]:
    """Pick the instance(s) the user referred to.

    For positional words, instances smaller than `min_rel_area` of the largest one
    are ignored first: people point at salient objects, not distant pedestrians.
    """
    if not detections:
        return []
    if which == "all":
        return list(detections)

    height, width = image_shape[:2]
    if which in ("left", "right", "top", "bottom", "center"):
        biggest = max(d.area for d in detections)
        detections = [d for d in detections if d.area >= min_rel_area * biggest]

    keys = {
        "best": lambda d: -d.confidence,
        "left": lambda d: d.center[0],
        "right": lambda d: -d.center[0],
        "top": lambda d: d.center[1],
        "bottom": lambda d: -d.center[1],
        "center": lambda d: (d.center[0] - width / 2) ** 2 + (d.center[1] - height / 2) ** 2,
        "largest": lambda d: -d.area,
        "smallest": lambda d: d.area,
    }
    return [min(detections, key=keys.get(which, keys["best"]))]
