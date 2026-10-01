"""Object removal by learned one-pass inpainting."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from editing.segment import binary_mask
from editing.utils import file_mb
from editing.weights import ensure_weights


class MiganInpainter:
    """MI-GAN 512 (Places2) as Picsart's exported ONNX pipeline.

    The exported graph crops around the hole, resizes to 512, inpaints, resizes back
    and blends. Its mask convention is 255 = keep, 0 = hole, so ours is inverted.
    """

    def __init__(self, weights: str | Path, threads: int = 0):
        import onnxruntime as ort

        path = ensure_weights(Path(weights))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = [x.name for x in self.session.get_inputs()]
        self.output = self.session.get_outputs()[0].name
        self.size_mb = file_mb(path)

    def __call__(self, image_bgr: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
        known = 255 - binary_mask(remove_mask)
        image = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[None].transpose(0, 3, 1, 2))
        mask = np.ascontiguousarray(known[None, None])
        result = self.session.run([self.output], {self.inputs[0]: image, self.inputs[1]: mask})[0]
        rgb = np.clip(result[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LamaInpainter:
    """big-lama (TorchScript). Quality reference: cleaner on large holes, 7x larger, 4-17x slower."""

    def __init__(self, weights: str | Path, context: float = 2.0):
        import torch

        path = ensure_weights(Path(weights))
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.context = context
        self.size_mb = file_mb(path)

    def __call__(self, image_bgr: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
        torch = self.torch
        mask = binary_mask(remove_mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return image_bgr.copy()
        height, width = mask.shape
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * self.context
        x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x2, y2 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))

        rgb = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        m = mask[y1:y2, x1:x2]
        h, w = m.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8  # LaMa needs sides divisible by 8
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        m = np.pad(m, ((0, ph), (0, pw)), mode="symmetric")

        image_t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        mask_t = (torch.from_numpy(m)[None, None] > 127).float()
        with torch.inference_mode():
            out = self.model(image_t, mask_t)[0].permute(1, 2, 0).numpy()[:h, :w]

        result = image_bgr.copy()
        result[y1:y2, x1:x2] = cv2.cvtColor(np.clip(out * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return result


def feather_blend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, feather: float = 7) -> np.ndarray:
    """Take `edited` inside the mask and `original` outside, with a soft edge."""
    alpha = cv2.GaussianBlur(binary_mask(mask), (0, 0), feather).astype(np.float32)[..., None] / 255.0
    out = alpha * edited.astype(np.float32) + (1 - alpha) * original.astype(np.float32)
    return out.astype(np.uint8)


def build_inpainter(backend: str, weights: str | Path, threads: int = 0):
    if backend == "migan":
        return MiganInpainter(weights, threads)
    if backend == "lama":
        return LamaInpainter(weights)
    raise ValueError(f"Unknown inpainting backend: {backend!r}")
