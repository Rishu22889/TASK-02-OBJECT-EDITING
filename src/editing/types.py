"""Data passed between pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


class ObjectNotFoundError(RuntimeError):
    """The instruction named an object that no detector found in the image."""


@dataclass
class ParsedInstruction:
    action: str  # "remove" or "replace"
    object: str  # free-text phrase, used by the open-vocabulary detector
    coco_class: str  # one of the 80 COCO names, or "none"
    which: str = "best"  # best | left | right | top | bottom | center | largest | smallest | all
    target: str = ""  # replacement object (replace only)
    target_prompt: str = ""  # diffusion prompt for the replacement (replace only)
    parser: str = "regex"
    raw: str = ""  # raw LLM output, for debugging

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("raw")
        return data


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    polygon: np.ndarray | None = None  # instance outline from a segmentation head, if any

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class EditResult:
    image: np.ndarray  # BGR output
    removal_mask: np.ndarray  # uint8, 255 = removed
    parsed: ParsedInstruction
    route: str  # which grounding path found the object
    insert_mode: str  # "", "diffusion" or "composite"
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage, plus "total"
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
