"""Instruction parsing: regex rules, a local LLM constrained by a JSON grammar, and a cascade of the two."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from editing.config import ParserConfig
from editing.types import ParsedInstruction
from editing.vocabulary import COCO_NAMES, COCO_SET, WHICH_OPTIONS, clean_phrase, to_coco_class

logger = logging.getLogger(__name__)


def default_target_prompt(target: str) -> str:
    return f"a photorealistic {target}, natural lighting, realistic shadow, sharp details, matching the scene"


def make_parsed(
    action: str,
    obj: str,
    target: str = "",
    which: str = "best",
    target_prompt: str = "",
    coco_class: str | None = None,
    parser: str = "regex",
    raw: str = "",
) -> ParsedInstruction:
    obj = clean_phrase(obj)
    target = clean_phrase(target) if action == "replace" else ""
    return ParsedInstruction(
        action=action,
        object=obj,
        coco_class=coco_class if coco_class in COCO_SET else to_coco_class(obj),
        which=which if which in WHICH_OPTIONS else "best",
        target=target,
        target_prompt=(target_prompt or default_target_prompt(target)) if action == "replace" else "",
        parser=parser,
        raw=raw,
    )


# --------------------------------------------------------------------------- regex

_REMOVE_VERBS = r"remove|delete|erase|eliminate|hide|get rid of|take out|take away|cut out|wipe out"
_REMOVE = (
    re.compile(rf"\b(?:{_REMOVE_VERBS})\s+(.+)"),
    re.compile(r"\bmake\s+(.+?)\s+(?:disappear|vanish|go away|invisible)\b"),
    re.compile(r"\b(?:want|need|like|have|get)\s+(.+?)\s+(?:removed|deleted|erased|gone|taken out)\b"),
    re.compile(r"\b(?:don't|dont|do not)\s+(?:want|need|like)\s+(.+)"),
)
# (pattern, object group, target group)
_REPLACE = (
    (re.compile(r"\b(?:replace|swap|exchange|switch)\s+(.+?)\s+(?:with|for|by)\s+(.+)"), 1, 2),
    (re.compile(r"\b(?:change|turn|convert|transform)\s+(.+?)\s+(?:into|to)\s+(.+)"), 1, 2),
    (re.compile(r"\b(?:want|need|like|have|get)\s+(.+?)\s+(?:replaced|swapped|changed|turned)\s+"
                r"(?:with|by|for|into|to)\s+(.+)"), 1, 2),
    (re.compile(r"(.+?)\s+(?:should|must|could)\s+(?:be|become)\s+(.+?)\s+instead\b"), 1, 2),
    (re.compile(r"\b(?:put|place|add|use|insert)\s+(.+?)\s+(?:instead of|in place of)\s+(.+)"), 2, 1),
)

# Instance words, longest first so "leftmost" is not read as "left".
_INSTANCE_WORDS = {
    "left": "left", "leftmost": "left", "left-most": "left", "left-hand": "left",
    "right": "right", "rightmost": "right", "right-most": "right", "right-hand": "right",
    "top": "top", "topmost": "top", "upper": "top",
    "bottom": "bottom", "bottommost": "bottom", "lower": "bottom",
    "middle": "center", "center": "center", "centre": "center", "central": "center",
    "biggest": "largest", "largest": "largest", "bigger": "largest", "larger": "largest",
    "smallest": "smallest", "tiniest": "smallest", "smaller": "smallest", "tinier": "smallest",
}
_INSTANCE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _INSTANCE_WORDS), key=len, reverse=True)) + r")\b")
_QUANTIFIER = re.compile(r"^(?:all|every|each|both)(?:\s+of)?(?:\s+(?:the|these|those|my|our))?\s+")
_ARTICLE = re.compile(r"^(?:the|a|an|that|this|those|these|my|your|our|some)\s+")

# Where the object phrase ends: a comma, a preposition, a relative clause or a
# participle after the noun ("man holding ...", but "hanging lamp" stays whole).
_PREPOSITIONS = (
    r"on|in|at|near|by|beside|behind|under|underneath|beneath|over|above|below|from|with|inside|"
    r"outside|along|across|around|against|between|next to|close to|towards?|to the"
)
_CLAUSE = r"that|who|which|where|whose|because|since|while"
_PARTICIPLES = (
    r"standing|sitting|walking|running|holding|wearing|carrying|riding|lying|laying|leaning|hanging|"
    r"parked|placed|located|positioned|playing|eating|looking|waiting|sleeping|flying|swimming|"
    r"driving|crossing|talking|posing|resting|jumping|climbing|perched|seated|stacked|floating|grazing"
)
_BOUNDARY = re.compile(rf"\s*,|\s+(?:{_PREPOSITIONS}|{_CLAUSE}|{_PARTICIPLES})\b")

_FILLER = re.compile(r"\s+(?:please|pls|now|anymore|any more|too|as well|for me|instead|thanks|thank you)$")
_SCENE = re.compile(
    r"\s+(?:in|from|on|of)\s+(?:the|this|my|our|that)\s+"
    r"(?:photo|image|picture|pic|scene|shot|frame|background|foreground)\b.*$"
)

# Plural-looking nouns that name one object, and plurals without a final "s".
_PLURAL_ONLY = frozenset({"glasses", "sunglasses", "pants", "jeans", "shorts", "trousers", "headphones"})
_IRREGULAR_PLURALS = frozenset({"people", "men", "women", "children", "mice", "geese"})


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _instance_words(text: str) -> list[str]:
    return [_INSTANCE_WORDS[w] for w in _INSTANCE.findall(text)]


def _tidy(span: str) -> str:
    """Drop end punctuation, politeness and "in the photo"-style tails."""
    span = re.sub(r"\bright (?:now|away)\b", " ", span)
    span = _SCENE.sub("", re.sub(r"\s+", " ", span).strip())
    previous = None
    while span != previous:
        previous = span
        span = _FILLER.sub("", re.sub(r"[.?!;:,]+$", "", span).strip())
    return span


def _is_plural(phrase: str) -> bool:
    words = re.findall(r"[a-z]+", phrase)
    if not words:
        return False
    head = words[-1]
    if head in _IRREGULAR_PLURALS:
        return True
    if head in COCO_SET or head in _PLURAL_ONLY or len(head) < 4:
        return False
    return head.endswith("s") and not head.endswith(("ss", "us", "is"))


def _object_and_which(span: str) -> tuple[str, str]:
    """Split "all of the people from the background" into ("people", "all")."""
    span = _tidy(span)
    found = _instance_words(span)
    which = found[0] if found else "best"
    quantifier = _QUANTIFIER.match(span)
    if quantifier:
        which, span = "all", span[quantifier.end():]
    span = _ARTICLE.sub("", span)
    words = _BOUNDARY.split(span, maxsplit=1)[0].split()
    # "the biggest car" -> "car"; a lone "top" stays the object.
    words = [w for w in words[:-1] if w not in _INSTANCE_WORDS] + words[-1:]
    obj = " ".join(words)
    if which == "best" and _is_plural(obj):
        which = "all"
    return obj, which


class RegexParser:
    """Rules: remove / replace phrasings, instance words and object-phrase trimming."""

    name = "regex"
    size_mb = 0.0

    def __call__(self, instruction: str) -> ParsedInstruction:
        text = _normalise(instruction)
        for pattern, obj_group, target_group in _REPLACE:
            m = pattern.search(text)
            if m:
                obj, which = _object_and_which(m.group(obj_group))
                return make_parsed("replace", obj, _tidy(m.group(target_group)), which, parser=self.name)
        for pattern in _REMOVE:
            m = pattern.search(text)
            if m:
                obj, which = _object_and_which(m.group(1))
                return make_parsed("remove", obj, which=which, parser=self.name)
        raise ValueError(f"Regex parser could not understand: {instruction!r}")


_VAGUE = frozenset({
    "thing", "things", "stuff", "object", "objects", "item", "items", "something", "anything",
    "everything", "it", "them", "this", "that", "one", "ones", "all", "whatever",
})
_TARGET_SEPARATOR = re.compile(r"\b(?:with|for|by|into|instead of|in place of)\b")
_NEGATION = re.compile(
    r"\b(?:don't|dont|do not|never|not)\s+(?:\w+\s+)?"
    r"(?:remove|delete|erase|eliminate|hide|replace|swap|change|turn|touch|edit|get rid)\b"
    r"|\b(?:don't|dont|do not)\s+want\b.*\b(?:removed|deleted|erased|gone|replaced|changed)\b"
    r"|\bnot\s+(?:the|that|this|those|these|a|an)\b"
    r"|\b(?:except|but not|other than|apart from|besides)\b"
)


def unusual(instruction: str, parsed: ParsedInstruction) -> bool:
    """True when a rules parse should be checked by the LLM.

    Flags a vague or long object phrase ("the thing", "red and white striped
    umbrella"), a long or ambiguous target ("the hat with a robot"), a negation
    ("don't remove the dog, ...") and conflicting instance words ("top left").
    """
    text = _normalise(instruction)
    words = parsed.object.split()
    return bool(
        not words
        or any(w in _VAGUE for w in words)
        or len(words) > 3
        or re.search(r",|\b(?:and|or)\b", parsed.object)
        or len(parsed.target.split()) > 4
        or _TARGET_SEPARATOR.search(parsed.target)
        or _NEGATION.search(text)
        or len(set(_instance_words(text))) > 1
    )


# --------------------------------------------------------------------------- LLM

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["remove", "replace"]},
        "object": {"type": "string"},
        "coco_class": {"type": "string", "enum": list(COCO_NAMES) + ["none"]},
        "which": {"type": "string", "enum": list(WHICH_OPTIONS)},
        "target": {"type": "string"},
        "target_prompt": {"type": "string"},
    },
    "required": ["action", "object", "coco_class", "which", "target", "target_prompt"],
}

SYSTEM_PROMPT = f"""You turn a photo-editing request into JSON with these keys:
- action: "remove" or "replace".
- object: the thing to edit as a short noun phrase for an object detector (1-3 words, singular, keep colour/material words, drop location words).
- coco_class: the closest class from this list, or "none" if nothing fits: {", ".join(COCO_NAMES)}.
  Map synonyms: guy/man/woman/kid/cyclist -> person, puppy -> dog, kitten -> cat, sedan/SUV -> car, bike -> bicycle, motorbike -> motorcycle.
- which: which instance: left, right, top, bottom, center, largest, smallest or all. Use "best" if the user does not say.
- target: for replace, the new object (1-4 words). Empty string for remove.
- target_prompt: for replace, a short photorealistic Stable Diffusion prompt for the new object. Empty string for remove.

Examples:
Request: erase the woman standing on the right
{{"action": "remove", "object": "woman", "coco_class": "person", "which": "right", "target": "", "target_prompt": ""}}
Request: change the sheep into a white horse
{{"action": "replace", "object": "sheep", "coco_class": "sheep", "which": "best", "target": "white horse", "target_prompt": "a photorealistic white horse standing, natural lighting, sharp details, realistic shadow"}}
Request: remove every bird from the sky
{{"action": "remove", "object": "bird", "coco_class": "bird", "which": "all", "target": "", "target_prompt": ""}}
Request: delete the lamp post
{{"action": "remove", "object": "lamp post", "coco_class": "none", "which": "best", "target": "", "target_prompt": ""}}
Answer with the JSON object only."""


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON in LLM output: {text[:200]!r}")
    return json.loads(m.group(0))


class LLMParser:
    """Qwen2.5-0.5B-Instruct (4-bit GGUF) via llama.cpp, with the JSON schema compiled to a grammar.

    Falls back to the same model in fp32 through `transformers` when
    llama-cpp-python is not installed (larger and slower; no grammar).
    """

    name = "llm"

    def __init__(self, cfg: ParserConfig, weights_dir: str | Path = "weights", threads: int = 0):
        self.cfg = cfg
        self.threads = threads or None
        self.size_mb = 0.0
        try:
            from llama_cpp import Llama  # noqa: F401

            self.backend = "llama_cpp"
            self._load_llama_cpp(Path(weights_dir) / "llm")
        except ImportError:
            self.backend = "transformers"
            self._load_transformers()

    def _load_llama_cpp(self, local_dir: Path) -> None:
        from huggingface_hub import hf_hub_download, list_repo_files
        from llama_cpp import Llama

        suffix = f"{self.cfg.gguf_quant.lower()}.gguf"
        local = sorted(local_dir.glob(f"*{suffix}")) if local_dir.exists() else []
        if local:
            path = local[0]
        else:
            name = next(f for f in list_repo_files(self.cfg.gguf_repo) if f.lower().endswith(suffix))
            path = Path(hf_hub_download(self.cfg.gguf_repo, name, local_dir=str(local_dir)))
        self.model_path = path
        self.llm = Llama(model_path=str(path), n_ctx=self.cfg.n_ctx, n_threads=self.threads, verbose=False)
        self.size_mb = path.stat().st_size / 2**20

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.warning("llama-cpp-python not installed: using the fp32 transformers backend.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_repo)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_repo, dtype=torch.float32).eval()
        self.size_mb = sum(p.numel() for p in self.model.parameters()) * 4 / 2**20

    def generate(self, instruction: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Request: {instruction}"},
        ]
        if self.backend == "llama_cpp":
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object", "schema": EDIT_SCHEMA},
                temperature=0.0,
                max_tokens=self.cfg.max_tokens,
            )
            return response["choices"][0]["message"]["content"]

        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def to_parsed(self, data: dict, raw: str = "", class_mapping: str | None = None) -> ParsedInstruction:
        """Turn the LLM's JSON into a ParsedInstruction.

        class_mapping "alias" maps the free-text `object` through the deterministic
        synonym table; a small model forced to pick from an 80-name enum otherwise
        drifts to the nearest legal prefix (kitten -> kite, SUV -> sports ball).
        """
        mapping = class_mapping or self.cfg.class_mapping
        action = data.get("action")
        if action not in ("remove", "replace"):
            action = "replace" if data.get("target") else "remove"
        obj = data.get("object", "")
        llm_class = str(data.get("coco_class", "none")).lower()
        coco = to_coco_class(obj) if mapping == "alias" else llm_class
        return make_parsed(
            action,
            obj,
            target=data.get("target", ""),
            which=str(data.get("which", "best")).lower(),
            target_prompt=data.get("target_prompt", ""),
            coco_class=coco if coco in COCO_SET else "none",
            parser=self.name,
            raw=raw,
        )

    def __call__(self, instruction: str) -> ParsedInstruction:
        raw = self.generate(instruction)
        return self.to_parsed(extract_json(raw), raw)


class FallbackParser:
    """Try the primary parser; on any failure use the regex rules."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback or RegexParser()
        self.name = getattr(primary, "name", "llm")
        self.size_mb = getattr(primary, "size_mb", 0.0)

    def __call__(self, instruction: str) -> ParsedInstruction:
        try:
            return self.primary(instruction)
        except Exception as exc:  # malformed JSON, missing fields, backend errors
            logger.warning("Primary parser failed (%s); using regex.", exc)
            return self.fallback(instruction)


class CascadeParser:
    """Rules first; the LLM only for instructions they cannot parse or find unusual.

    `load_llm` is called on the first such instruction. If it fails (for example
    no LLM backend is installed) the rules parse is kept, and the load is not retried.
    """

    name = "cascade"
    size_mb = 0.0  # the LLM's weights are reported by whoever loads it

    def __init__(self, load_llm: Callable[[], Callable[[str], ParsedInstruction]], rules=None):
        self.rules = rules or RegexParser()
        self._load_llm = load_llm
        self._llm = None
        self._llm_failed = False

    def _get_llm(self):
        if self._llm is None and not self._llm_failed:
            try:
                self._llm = self._load_llm()
            except Exception as exc:  # missing backend, download or load errors
                self._llm_failed = True
                logger.warning("LLM parser unavailable (%s); using the regex rules only.", exc)
        return self._llm

    def __call__(self, instruction: str) -> ParsedInstruction:
        try:
            parsed, error = self.rules(instruction), None
        except ValueError as exc:
            parsed, error = None, exc
        if parsed is not None and not unusual(instruction, parsed):
            return parsed
        llm = self._get_llm()
        if llm is not None:
            try:
                return llm(instruction)
            except Exception as exc:  # malformed JSON, missing fields, backend errors
                logger.warning("LLM parser failed (%s); using the regex rules.", exc)
        if parsed is None:
            raise error
        return parsed


def build_parser(
    cfg: ParserConfig,
    weights_dir: str | Path = "weights",
    threads: int = 0,
    load_llm: Callable[[], Callable[[str], ParsedInstruction]] | None = None,
):
    """`load_llm` builds the cascade's LLM on demand (default: a new LLMParser)."""
    if cfg.backend == "regex":
        return RegexParser()
    if cfg.backend == "cascade":
        return CascadeParser(load_llm or (lambda: LLMParser(cfg, weights_dir, threads)))
    if cfg.backend == "llm":
        return FallbackParser(LLMParser(cfg, weights_dir, threads))
    raise ValueError(f"Unknown parser backend: {cfg.backend!r}")
