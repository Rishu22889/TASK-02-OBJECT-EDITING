"""The end-to-end editor: parse -> ground -> segment -> remove -> (insert)."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np

from editing.config import EditorConfig
from editing.ground import CocoDetector, OpenVocabDetector, select_detections
from editing.inpaint import build_inpainter
from editing.insert import CompositeInserter, DiffusionInserter
from editing.parse import LLMParser, build_parser
from editing.segment import MobileSamSegmenter, build_removal_mask, polygon_to_mask
from editing.types import Detection, EditResult, ObjectNotFoundError, ParsedInstruction
from editing.utils import configure_threads

logger = logging.getLogger(__name__)

STAGES = ("parse", "detect", "segment", "remove", "insert")


class Editor:
    """Text-driven object removal and replacement.

    Components load lazily on first use, so a removal-only workload never loads
    the LLM, YOLO-World, MobileSAM or Stable Diffusion unless it needs them:
    the LLM parses only instructions the regex rules find unusual, COCO objects
    take the fast path (YOLOv8n-seg box + mask in one pass), and YOLO-World +
    MobileSAM run only when the class is "none" or nothing is found.
    Model loading is timed separately ("load") and never counted in a stage.
    """

    def __init__(self, config: EditorConfig | None = None):
        self.cfg = config or EditorConfig()
        self.threads = configure_threads(self.cfg.threads)
        self._components: dict[str, object] = {}
        self.load_seconds: dict[str, float] = {}

    # ------------------------------------------------------------ components

    def _get(self, name: str, factory: Callable[[], object]):
        if name not in self._components:
            start = time.perf_counter()
            self._components[name] = factory()
            self.load_seconds[name] = time.perf_counter() - start
            logger.info("loaded %s in %.1f s", name, self.load_seconds[name])
        return self._components[name]

    @property
    def parser(self):
        return self._get("parser", lambda: build_parser(
            self.cfg.parser, self.cfg.weights_dir, self.threads, load_llm=lambda: self.llm_parser))

    @property
    def llm_parser(self) -> LLMParser:
        """Loaded on demand by the cascade parser, so its load time and size are reported on their own."""
        return self._get("llm_parser", lambda: LLMParser(self.cfg.parser, self.cfg.weights_dir, self.threads))

    @property
    def coco(self) -> CocoDetector:
        g = self.cfg.grounding
        return self._get("coco_detector", lambda: CocoDetector(
            self.cfg.weight_path(g.coco_weights), g.conf, g.imgsz, self.cfg.device))

    @property
    def open_vocab(self) -> OpenVocabDetector:
        g = self.cfg.grounding
        return self._get("open_vocab_detector", lambda: OpenVocabDetector(
            self.cfg.weight_path(g.open_vocab_weights), g.open_vocab_conf, g.open_vocab_min_conf,
            g.imgsz, self.cfg.device))

    @property
    def segmenter(self) -> MobileSamSegmenter:
        return self._get("mobile_sam", lambda: MobileSamSegmenter(
            self.cfg.weight_path(self.cfg.mask.mobile_sam_weights), self.cfg.device))

    @property
    def inpainter(self):
        c = self.cfg.inpaint
        weights = c.migan_weights if c.backend == "migan" else c.lama_weights
        return self._get("inpainter", lambda: build_inpainter(c.backend, self.cfg.weight_path(weights), self.threads))

    @property
    def diffusion(self) -> DiffusionInserter:
        return self._get("diffusion", lambda: DiffusionInserter(self.cfg.insert))

    @property
    def composite(self) -> CompositeInserter:
        return self._get("asset_library", lambda: CompositeInserter(
            self.cfg.insert.asset_dir, lambda image, boxes: self.segmenter.segment(image, boxes)))

    def footprint(self) -> dict[str, float]:
        """Weights (MB) of every component loaded so far."""
        return {name: round(getattr(c, "size_mb", 0.0), 2) for name, c in self._components.items()}

    # ------------------------------------------------------------ stages

    def parse(self, instruction: str) -> ParsedInstruction:
        return self.parser(instruction)

    def ground(self, image: np.ndarray, parsed: ParsedInstruction) -> tuple[list[Detection], str]:
        """Fast COCO path first; open vocabulary only for "none" classes or when nothing is found."""
        g = self.cfg.grounding
        if parsed.coco_class != "none":
            found = select_detections(self.coco.detect(image, parsed.coco_class), parsed.which, image.shape, g.min_rel_area)
            if found:
                return found, f"coco:{Path(g.coco_weights).stem}"
        if g.open_vocab:
            phrase = parsed.object or parsed.coco_class
            found = select_detections(self.open_vocab.detect(image, phrase), parsed.which, image.shape, g.min_rel_area)
            if found:
                return found, "open-vocab:yolo-world+mobilesam"
        return [], "none"

    def masks(self, image: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        if self.cfg.mask.segmenter == "seg" and all(d.polygon is not None for d in detections):
            return [polygon_to_mask(d.polygon, image.shape) for d in detections]
        return self.segmenter.segment(image, [d.bbox for d in detections])

    def insert(self, background: np.ndarray, bbox: tuple, parsed: ParsedInstruction, mode: str) -> tuple[np.ndarray, str]:
        if mode == "composite":
            asset = self.composite.find_asset(parsed.target)
            if asset is not None:
                return self.composite.insert(background, bbox, asset), "composite"
            if not self.cfg.insert.composite_fallback:
                raise ValueError(f"No asset for {parsed.target!r} in {self.cfg.insert.asset_dir}/")
            logger.info("No asset for %r; falling back to diffusion.", parsed.target)
        negative = f"{parsed.object}, {self.cfg.insert.negative_prompt}"
        return self.diffusion.generate(background, bbox, parsed.target_prompt, negative), "diffusion"

    # ------------------------------------------------------------ end to end

    @contextmanager
    def _stage(self, timings: dict[str, float], key: str):
        """Time a stage, moving any model loading it triggered into timings["load"]."""
        loaded_before = sum(self.load_seconds.values())
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            loaded = sum(self.load_seconds.values()) - loaded_before
            timings[key] = timings.get(key, 0.0) + elapsed - loaded
            if loaded:
                timings["load"] = timings.get("load", 0.0) + loaded

    def edit(self, image: np.ndarray, instruction: str, mode: str | None = None) -> EditResult:
        """Apply one instruction to a BGR image. `mode` overrides the configured insert mode.

        `timings` holds seconds per stage and their "total"; first-use model loading
        is reported apart as "load".
        """
        timings: dict[str, float] = {}
        with self._stage(timings, "parse"):
            parsed = self.parse(instruction)
        with self._stage(timings, "detect"):
            detections, route = self.ground(image, parsed)
        if not detections:
            raise ObjectNotFoundError(f"Could not find {parsed.object or parsed.coco_class!r} in the image")
        boxes = [d.bbox for d in detections]

        with self._stage(timings, "segment"):
            masks = self.masks(image, detections)
        m = self.cfg.mask
        removal = build_removal_mask(masks, boxes, image.shape, m.grow, m.shadow_ratio)
        with self._stage(timings, "remove"):
            output = self.inpainter(image, removal)

        insert_mode = ""
        if parsed.action == "replace":
            with self._stage(timings, "insert"):
                output, insert_mode = self.insert(output, boxes[0], parsed, mode or self.cfg.insert.mode)

        timings["total"] = sum(timings.get(s, 0.0) for s in STAGES)
        return EditResult(output, removal, parsed, route, insert_mode, timings, boxes)
