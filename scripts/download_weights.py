"""Download the weights a deployment profile needs, and load each model once to verify it.

    python scripts/download_weights.py --profile core        # YOLOv8n-seg + MI-GAN           (~34 MB)
    python scripts/download_weights.py --profile nl          # + Qwen2.5-0.5B Q4 parser        (~0.5 GB)
    python scripts/download_weights.py --profile open-vocab  # + YOLO-World, CLIP text, SAM    (~0.9 GB)
    python scripts/download_weights.py --profile full        # + SD-1.5 inpaint, LCM, TAESD   (~4.7 GB)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editing import Editor, load_config  # noqa: E402
from editing.weights import ensure_weights  # noqa: E402

PROFILES = {
    "core": ["coco", "inpainter"],
    "nl": ["coco", "inpainter", "parser"],
    "open-vocab": ["coco", "inpainter", "parser", "open_vocab", "segmenter"],
    "full": ["coco", "inpainter", "parser", "open_vocab", "segmenter", "diffusion"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, default="full")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--lama", action="store_true", help="also fetch big-lama (quality reference, 196 MB)")
    args = ap.parse_args()

    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = {"parser": {"backend": "llm"}} if "parser" in PROFILES[args.profile] else {}
    cfg = load_config(args.config, overrides)
    editor = Editor(cfg)

    for component in PROFILES[args.profile]:
        print(f"loading {component} ...", flush=True)
        getattr(editor, component)
    if args.lama:
        ensure_weights(cfg.weight_path(cfg.inpaint.lama_weights))

    print(json.dumps({"profile": args.profile, "weights_mb": editor.footprint()}, indent=2))


if __name__ == "__main__":
    main()
