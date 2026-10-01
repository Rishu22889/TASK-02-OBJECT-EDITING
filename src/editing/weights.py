"""Locating and downloading model weights."""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Single-file weights that are fetched by URL. Ultralytics, Hugging Face and
# CLIP weights are fetched by their own libraries (see `ensure_weights`).
WEIGHT_URLS: dict[str, str] = {
    "mobile_sam.pt": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    "migan_pipeline_v2.onnx": "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
    "big-lama.pt": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
}


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest` atomically (via a .part file). Skips existing files."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.replace(dest)
    return dest


def ensure_weights(path: Path) -> Path:
    """Make sure a weight file exists locally, downloading it if its name is known."""
    path = Path(path)
    if path.exists():
        return path
    if path.name in WEIGHT_URLS:
        return download(WEIGHT_URLS[path.name], path)
    if path.suffix == ".pt":  # an Ultralytics asset name such as yolov8n-seg.pt
        from ultralytics.utils.downloads import attempt_download_asset

        path.parent.mkdir(parents=True, exist_ok=True)
        return Path(attempt_download_asset(str(path)))
    raise FileNotFoundError(f"Missing weights and no known download source: {path}")


def find_clip_text_encoders() -> list[Path]:
    """CLIP checkpoints that YOLO-World downloads on its first `set_classes` call.

    Older Ultralytics releases cache them in ~/.cache/clip, newer ones in
    <Ultralytics weights_dir>/clip; MobileCLIP TorchScript files sit in weights_dir.
    """
    roots = [Path.home() / ".cache" / "clip", Path.cwd()]
    try:
        from ultralytics.utils import WEIGHTS_DIR

        roots += [Path(WEIGHTS_DIR) / "clip", Path(WEIGHTS_DIR)]
    except ImportError:
        pass
    found: set[Path] = set()
    for root in roots:
        if root.exists():
            found.update(p.resolve() for p in root.glob("*.pt") if p.name.lower().startswith(("vit-", "rn")))
            found.update(p.resolve() for p in root.glob("mobileclip*.ts"))
    return sorted(found)
