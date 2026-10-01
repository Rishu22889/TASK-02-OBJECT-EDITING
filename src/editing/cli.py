"""Command line: edit one image with one instruction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from editing.config import load_config
from editing.pipeline import Editor
from editing.types import ObjectNotFoundError
from editing.utils import read_image, write_image


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m editing",
        description="Remove or replace an object in an image from a plain-English instruction (CPU only).",
    )
    ap.add_argument("image", help="input image")
    ap.add_argument("instruction", help='e.g. "remove the car" or "replace the cat with a dog"')
    ap.add_argument("-o", "--output", default="outputs/edited.jpg", help="output image path")
    ap.add_argument("-c", "--config", default="configs/default.yaml", help="YAML config")
    ap.add_argument("--parser", choices=["cascade", "llm", "regex"], help="override the parser backend")
    ap.add_argument("--mode", choices=["diffusion", "composite"], help="override the replacement mode")
    ap.add_argument("--no-open-vocab", action="store_true", help="COCO classes only")
    ap.add_argument("--threads", type=int, help="CPU threads (0 = all)")
    ap.add_argument("--save-mask", action="store_true", help="also write the removal mask")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    overrides: dict = {}
    if args.parser:
        overrides["parser"] = {"backend": args.parser}
    if args.no_open_vocab:
        overrides["grounding"] = {"open_vocab": False}
    if args.threads is not None:
        overrides["threads"] = args.threads
    config_path = args.config if Path(args.config).exists() else None
    editor = Editor(load_config(config_path, overrides))

    try:
        result = editor.edit(read_image(args.image), args.instruction, mode=args.mode)
    except (ObjectNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = write_image(args.output, result.image)
    report = {
        "output": str(out),
        "parsed": result.parsed.to_dict(),
        "route": result.route,
        "insert_mode": result.insert_mode,
        "timings_s": {k: round(v, 3) for k, v in result.timings.items()},
        "weights_mb": editor.footprint(),
    }
    if args.save_mask:
        report["mask"] = str(write_image(out.with_name(out.stem + "_mask.png"), result.removal_mask))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
