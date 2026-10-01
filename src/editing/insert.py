"""Inserting the replacement object into the cleaned background."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from editing.config import InsertConfig
from editing.inpaint import feather_blend
from editing.segment import placement_mask
from editing.utils import file_mb, fp32_mb

logger = logging.getLogger(__name__)


def square_crop_box(bbox: tuple, image_shape: tuple, padding_ratio: float = 0.35) -> tuple[int, int, int, int]:
    """Square context crop around the object; SD-1.5 is trained on square 512 px images."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    side = int(max(x2 - x1, y2 - y1) * (1 + 2 * padding_ratio))
    side = max(8, min(side, h, w))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left = int(np.clip(cx - side // 2, 0, w - side))
    top = int(np.clip(cy - side // 2, 0, h - side))
    return left, top, left + side, top + side


class DiffusionInserter:
    """SD-1.5 inpainting with LCM-LoRA fused into the UNet (4 steps, no CFG) and the TAESD tiny VAE."""

    def __init__(self, cfg: InsertConfig):
        import torch
        from diffusers import (
            AutoencoderTiny,
            DPMSolverMultistepScheduler,
            LCMScheduler,
            StableDiffusionInpaintPipeline,
        )

        self.torch = torch
        self.cfg = cfg
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg.sd_model, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
        )
        scheduler_config = pipe.scheduler.config
        if cfg.lcm_lora:
            pipe.load_lora_weights(cfg.lcm_lora, adapter_name="lcm")
            pipe.fuse_lora()
            try:
                pipe.unload_lora_weights()  # weights stay fused into the UNet
            except Exception as exc:
                logger.debug("unload_lora_weights skipped: %s", exc)
            pipe.scheduler = LCMScheduler.from_config(scheduler_config)
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.full_vae = pipe.vae
        self.tiny_vae = self._load_tiny_vae(AutoencoderTiny) if cfg.tiny_vae else None

        vae = self.tiny_vae if self.tiny_vae is not None else self.full_vae
        self.size_mb = fp32_mb(pipe.unet) + fp32_mb(pipe.text_encoder) + fp32_mb(vae)

    def _load_tiny_vae(self, autoencoder_cls):
        tiny = autoencoder_cls.from_pretrained(self.cfg.tiny_vae, torch_dtype=self.torch.float32)
        self.pipe.vae = tiny
        try:  # smoke test: the inpaint pipeline must be able to encode and decode with it
            self._run(Image.new("RGB", (256, 256), (128, 128, 128)), Image.new("L", (256, 256), 255), "test", 1, 256)
            return tiny
        except Exception as exc:
            logger.warning("Tiny VAE unusable with this diffusers version (%s); keeping the full VAE.", exc)
            self.pipe.vae = self.full_vae
            return None

    def _run(self, image: Image.Image, mask: Image.Image, prompt: str, steps: int, size: int, negative: str = ""):
        kwargs = dict(
            prompt=prompt, image=image, mask_image=mask, height=size, width=size,
            num_inference_steps=steps, guidance_scale=self.cfg.guidance, strength=1.0,
            generator=self.torch.Generator("cpu").manual_seed(self.cfg.seed),
        )
        if self.cfg.guidance > 1.0 and negative:  # a negative prompt only acts with CFG
            kwargs["negative_prompt"] = negative
        with self.torch.inference_mode():
            return self.pipe(**kwargs).images[0]

    def generate(self, background: np.ndarray, bbox: tuple, prompt: str, negative_prompt: str = "") -> np.ndarray:
        """Generate the new object inside a box-shaped placement mask on a square crop, then blend back."""
        place = placement_mask(bbox, background.shape)
        x1, y1, x2, y2 = square_crop_box(bbox, background.shape)
        size = self.cfg.size
        crop = Image.fromarray(cv2.cvtColor(background[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        crop = crop.resize((size, size), Image.Resampling.LANCZOS)
        mask = Image.fromarray(place[y1:y2, x1:x2]).resize((size, size), Image.Resampling.NEAREST)

        generated = self._run(crop, mask, prompt, self.cfg.steps, size, negative_prompt or self.cfg.negative_prompt)
        generated = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
        edited = background.copy()
        edited[y1:y2, x1:x2] = cv2.resize(generated, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
        return feather_blend(background, edited, place, feather=5)


class CompositeInserter:
    """Fast mode: paste a stored cutout, colour-matched, with a soft contact shadow. No diffusion."""

    ALIASES = {"puppy": "dog", "doggy": "dog", "bike": "bicycle", "cycle": "bicycle",
               "automobile": "car", "sedan": "car"}

    def __init__(self, asset_dir: str | Path, segment_fn: Callable | None = None):
        self.asset_dir = Path(asset_dir)
        self.segment_fn = segment_fn  # (image, boxes) -> masks, for assets without transparency
        self._cache: dict[Path, np.ndarray] = {}
        self.size_mb = file_mb(self.asset_dir) if self.asset_dir.exists() else 0.0

    def find_asset(self, target: str) -> Path | None:
        """assets/<name>.* or assets/<name>_*.* for the head noun of the target ("red bicycle" -> bicycle)."""
        for word in reversed(re.findall(r"[a-z]+", str(target).lower())):
            name = self.ALIASES.get(word, word)
            for pattern in (f"{name}.*", f"{name}_*.*"):
                hits = sorted(self.asset_dir.glob(pattern))
                if hits:
                    return hits[0]
        return None

    def load_rgba(self, path: Path) -> np.ndarray:
        """BGRA cutout, tightly cropped. Uses the file's alpha, else segments the main object."""
        if path in self._cache:
            return self._cache[path]
        bgra = np.ascontiguousarray(np.array(Image.open(path).convert("RGBA"))[..., [2, 1, 0, 3]])
        if (bgra[..., 3] < 250).mean() < 0.01:  # no real transparency
            h, w = bgra.shape[:2]
            if self.segment_fn is not None:
                box = [w * 0.02, h * 0.02, w * 0.98, h * 0.98]
                bgra[..., 3] = self.segment_fn(np.ascontiguousarray(bgra[..., :3]), [box])[0]
            else:  # treat a near-white background as transparent
                bgra[..., 3] = np.where((bgra[..., :3] > 240).all(axis=2), 0, 255).astype(np.uint8)
        ys, xs = np.where(bgra[..., 3] > 10)
        bgra = np.ascontiguousarray(bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        self._cache[path] = bgra
        return bgra

    def insert(self, background: np.ndarray, bbox: tuple, asset: Path, fill: float = 0.95,
               shadow_strength: float = 0.45) -> np.ndarray:
        rgba = self.load_rgba(asset)
        h, w = background.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        scale = fill * min(bw / rgba.shape[1], bh / rgba.shape[0])
        ow, oh = max(1, int(rgba.shape[1] * scale)), max(1, int(rgba.shape[0] * scale))
        obj = cv2.resize(rgba, (ow, oh), interpolation=cv2.INTER_AREA)
        ox = int(np.clip(x1 + (bw - ow) // 2, 0, w - ow))
        oy = int(np.clip(y2 - oh, 0, h - oh))  # bottom-aligned: objects stand on the ground

        out = background.astype(np.float32)
        if shadow_strength > 0:
            shadow = np.zeros((h, w), np.float32)
            cv2.ellipse(shadow, (ox + ow // 2, min(h - 1, oy + oh - 2)),
                        (max(2, int(ow * 0.45)), max(2, oh // 14)), 0, 0, 360, 1.0, -1)
            shadow = cv2.GaussianBlur(shadow, (0, 0), max(2.0, oh / 30))
            out *= (1 - shadow_strength * shadow)[..., None]

        rgb = match_color(obj[..., :3], background[oy:oy + oh, ox:ox + ow], alpha=obj[..., 3]).astype(np.float32)
        a = cv2.GaussianBlur(obj[..., 3], (3, 3), 0).astype(np.float32)[..., None] / 255.0
        roi = out[oy:oy + oh, ox:ox + ow]
        out[oy:oy + oh, ox:ox + ow] = a * rgb + (1 - a) * roi
        return np.clip(out, 0, 255).astype(np.uint8)


def match_color(fg_bgr: np.ndarray, bg_bgr: np.ndarray, alpha: np.ndarray | None = None,
                strength: tuple[float, float, float] = (0.5, 0.15, 0.15)) -> np.ndarray:
    """Partial LAB mean/std transfer toward the local background (mostly brightness)."""
    fg = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = alpha > 10 if alpha is not None else np.ones(fg.shape[:2], bool)
    if sel.sum() < 10:
        return fg_bgr
    for c in range(3):
        f_mean, f_std = fg[..., c][sel].mean(), fg[..., c][sel].std() + 1e-6
        b_mean, b_std = bg[..., c].mean(), bg[..., c].std() + 1e-6
        k = strength[c]
        fg[..., c] = (fg[..., c] - f_mean) * (1 + k * (b_std / f_std - 1)) + f_mean + k * (b_mean - f_mean)
    return cv2.cvtColor(np.clip(fg, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
