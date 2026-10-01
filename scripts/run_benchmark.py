"""Benchmark the pipeline on a case set: per-stage CPU latency, peak RSS and model footprint.

    python scripts/run_benchmark.py                                  # final cases, configs/default.yaml
    python scripts/run_benchmark.py --config configs/core_removal.yaml --set core --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editing import Editor, load_config  # noqa: E402
from editing.utils import PeakRSS, read_image, save_json, system_info, write_image  # noqa: E402

STAGES = ("parse", "detect", "segment", "remove", "insert", "total", "load")


def slug(text: str, n: int = 40) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cases", default="data/benchmark_cases.json")
    ap.add_argument("--set", default="final", help="key in the cases file")
    ap.add_argument("--images", default="data/samples")
    ap.add_argument("--out", default="benchmarks/final_benchmark.csv")
    ap.add_argument("--save-dir", default="outputs/benchmark", help="where edited images go ('' = none)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs of the first case")
    ap.add_argument("--repeats", type=int, default=1, help="timed runs per case")
    ap.add_argument("--mode", choices=["diffusion", "composite"])
    args = ap.parse_args()

    os.chdir(ROOT)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    editor = Editor(load_config(args.config))
    cases = json.loads(Path(args.cases).read_text())[args.set]

    first = read_image(Path(args.images) / cases[0]["image"])
    for _ in range(args.warmup):
        try:
            editor.edit(first, cases[0]["instruction"], mode=args.mode)
        except Exception as exc:
            print(f"warm-up failed: {exc}")

    rows = []
    for case in cases:
        image = read_image(Path(args.images) / case["image"])
        for rep in range(args.repeats):
            row = {"image": case["image"], "instruction": case["instruction"], "repeat": rep, "error": ""}
            try:
                with PeakRSS() as mem:
                    result = editor.edit(image, case["instruction"], mode=args.mode)
            except Exception as exc:
                row["error"] = str(exc)[:160]
                rows.append(row)
                print(f"FAIL  {case['image']:<12} {case['instruction']} -> {exc}")
                break
            p = result.parsed
            row.update(action=p.action, object=p.object, coco_class=p.coco_class, which=p.which,
                       target=p.target, parser=p.parser, route=result.route, insert_mode=result.insert_mode)
            row.update({f"{s}_sec": round(result.timings.get(s, 0.0), 4) for s in STAGES})
            row.update(peak_rss_mb=round(mem.peak_mb, 1), peak_rss_delta_mb=round(mem.delta_mb, 1))
            rows.append(row)
            print(f"ok    {case['image']:<12} {case['instruction'][:48]:<48} {result.route:<32} "
                  f"{result.timings['total']:7.2f} s")
            if args.save_dir and rep == 0:
                write_image(Path(args.save_dir) / f"{Path(case['image']).stem}_{slug(case['instruction'])}.jpg",
                            result.image)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for action in ("remove", "replace"):
        ok = [r for r in rows if r.get("action") == action and not r["error"]]
        if ok:
            summary[action] = {f"{s}_sec": round(statistics.mean(r[f"{s}_sec"] for r in ok), 3) for s in STAGES[:-1]}
            summary[action]["cases"] = len(ok)
    summary["failures"] = sum(1 for r in rows if r["error"])
    meta = {"system": system_info(editor.threads), "weights_mb": editor.footprint(),
            "load_sec": {k: round(v, 2) for k, v in editor.load_seconds.items()}, "summary": summary}
    save_json(out.with_suffix(".json"), meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
