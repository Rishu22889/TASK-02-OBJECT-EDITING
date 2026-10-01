"""Score instruction parsers on the 26-instruction test set.

Each LLM output is scored twice: with the class label the LLM picked from the
80-name enum ("llm"), and with its free-text object mapped through the alias
table ("alias"). "cascade" (the default in configs/default.yaml) keeps the regex
parse unless the rules find the instruction unusual, then takes the alias LLM parse.

    python scripts/evaluate_parser.py            # regex only (no model needed)
    python scripts/evaluate_parser.py --llm      # regex + LLM + cascade (downloads ~0.5 GB once)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editing.config import load_config  # noqa: E402
from editing.parse import LLMParser, RegexParser, extract_json, unusual  # noqa: E402

CHECKS = ("action_ok", "class_ok", "object_ok", "which_ok", "target_ok", "all_ok")


def score(parsed, case: dict) -> dict:
    if parsed is None:
        return {k: False for k in CHECKS}
    ok = {
        "action_ok": parsed.action == case["action"],
        "class_ok": parsed.coco_class in case["coco"],
        "object_ok": any(k in parsed.object for k in case["object_keywords"]),
        "which_ok": parsed.which == case["which"],
        "target_ok": case["target"] is None or case["target"] in parsed.target,
    }
    ok["all_ok"] = all(ok.values())
    return ok


def row(variant: str, case: dict, parsed, seconds: float, error: str = "") -> dict:
    fields = parsed.to_dict() if parsed is not None else {}
    return {"parser": variant, "instruction": case["instruction"],
            **{k: fields.get(k, "") for k in ("action", "object", "coco_class", "which", "target")},
            **score(parsed, case), "latency_ms": round(seconds * 1000, 2), "error": error}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="data/parser_testset.json")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM parser")
    ap.add_argument("--out", default="benchmarks/parser_eval.csv")
    args = ap.parse_args()

    os.chdir(ROOT)
    cases = json.loads(Path(args.testset).read_text())
    rows = []

    regex, confident = RegexParser(), {}
    for case in cases:
        start = time.perf_counter()
        try:
            parsed, error = regex(case["instruction"]), ""
        except ValueError as exc:
            parsed, error = None, str(exc)
        seconds = time.perf_counter() - start
        rows.append(row("regex", case, parsed, seconds, error))
        if parsed is not None and not unusual(case["instruction"], parsed):
            confident[case["instruction"]] = (parsed, seconds)

    if args.llm:
        llm = LLMParser(load_config(args.config).parser, "weights")
        llm.generate("remove the car")  # warm-up
        for case in cases:
            start = time.perf_counter()
            raw = llm.generate(case["instruction"])
            seconds = time.perf_counter() - start
            for mapping in ("llm", "alias"):
                try:
                    parsed, error = llm.to_parsed(extract_json(raw), raw, class_mapping=mapping), ""
                except Exception as exc:
                    parsed, error = None, str(exc)
                rows.append(row(f"llm[{mapping}]", case, parsed, seconds, error))
            if case["instruction"] in confident:
                rows.append(row("cascade", case, *confident[case["instruction"]]))
            else:
                rows.append({**rows[-1], "parser": "cascade"})  # the alias LLM parse

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'parser':<12}" + "".join(f"{c[:-3]:>9}" for c in CHECKS) + f"{'ms':>10}")
    for variant in dict.fromkeys(r["parser"] for r in rows):
        sub = [r for r in rows if r["parser"] == variant]
        accs = "".join(f"{100 * statistics.mean(r[c] for r in sub):>8.0f}%" for c in CHECKS)
        print(f"{variant:<12}{accs}{statistics.mean(r['latency_ms'] for r in sub):>10.1f}")
    print(f"\nregex confident on {len(confident)}/{len(cases)}; the cascade sends the rest to the LLM")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
