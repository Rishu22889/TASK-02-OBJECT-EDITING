"""Pipeline configuration: nested dataclasses, loadable from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParserConfig:
    backend: str = "cascade"  # "cascade" (regex first, LLM for unusual phrasing), "llm" (regex fallback) or "regex"
    class_mapping: str = "alias"  # "alias": map the LLM's free-text object; "llm": trust its class label
    gguf_repo: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    gguf_quant: str = "q4_k_m"
    hf_repo: str = "Qwen/Qwen2.5-0.5B-Instruct"  # transformers fallback if llama-cpp-python is missing
    n_ctx: int = 2048
    max_tokens: int = 200


@dataclass
class GroundingConfig:
    coco_weights: str = "yolov8n-seg.pt"
    conf: float = 0.25
    imgsz: int = 640
    open_vocab: bool = True
    open_vocab_weights: str = "yolov8s-worldv2.pt"
    open_vocab_conf: float = 0.15
    open_vocab_min_conf: float = 0.05
    min_rel_area: float = 0.2  # positional selection ignores instances smaller than this share of the largest


@dataclass
class MaskConfig:
    segmenter: str = "seg"  # "seg": masks from the YOLOv8n-seg pass; "mobilesam": MobileSAM on every box
    mobile_sam_weights: str = "mobile_sam.pt"
    grow: int = 15  # 3x3 dilation iterations around the object mask
    shadow_ratio: float = 0.15  # extend the mask below the box to cover the contact shadow


@dataclass
class InpaintConfig:
    backend: str = "migan"  # "migan" or "lama"
    migan_weights: str = "migan_pipeline_v2.onnx"
    lama_weights: str = "big-lama.pt"


@dataclass
class InsertConfig:
    mode: str = "diffusion"  # "diffusion" (any text target) or "composite" (asset library)
    composite_fallback: bool = True  # composite mode: use diffusion when the target has no asset
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    lcm_lora: str = "latent-consistency/lcm-lora-sdv1-5"  # "" = plain SD with DPM-Solver
    tiny_vae: str = "madebyollin/taesd"  # "" = keep the full VAE
    steps: int = 4
    guidance: float = 1.0
    size: int = 512
    seed: int = 42
    negative_prompt: str = (
        "blurry, deformed, distorted, duplicate, extra limbs, cartoon, painting, "
        "illustration, low quality, text, watermark"
    )
    asset_dir: str = "assets"


@dataclass
class EditorConfig:
    weights_dir: str = "weights"
    threads: int = 0  # 0 = all logical cores
    device: str = "cpu"
    parser: ParserConfig = field(default_factory=ParserConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    insert: InsertConfig = field(default_factory=InsertConfig)

    def weight_path(self, name: str) -> Path:
        """Absolute or existing paths are used as given; bare names live in `weights_dir`."""
        path = Path(name)
        if path.is_absolute() or path.exists():
            return path
        return Path(self.weights_dir) / name


def _merge(obj: Any, values: dict) -> Any:
    known = {f.name for f in fields(obj)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {type(obj).__name__}: {sorted(unknown)}")
    for name, value in values.items():
        current = getattr(obj, name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, name, value)
    return obj


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> EditorConfig:
    """Defaults, then the YAML file, then `overrides` (same nested structure)."""
    cfg = EditorConfig()
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg
