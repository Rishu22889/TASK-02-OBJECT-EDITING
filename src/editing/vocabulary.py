"""Detector vocabulary: COCO class names, synonyms and phrase normalisation."""

from __future__ import annotations

import re

# Ultralytics / COCO order (class id = index).
COCO_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
COCO_SET = frozenset(COCO_NAMES)

WHICH_OPTIONS: tuple[str, ...] = (
    "best", "left", "right", "top", "bottom", "center", "largest", "smallest", "all",
)

ALIASES: dict[str, str] = {
    "bike": "bicycle", "cycle": "bicycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "guy": "person", "man": "person", "men": "person", "woman": "person", "women": "person",
    "people": "person", "persons": "person", "kid": "person", "child": "person", "boy": "person",
    "girl": "person", "cyclist": "person", "rider": "person", "pedestrian": "person",
    "lady": "person", "human": "person",
    "puppy": "dog", "doggy": "dog", "kitten": "cat", "kitty": "cat",
    "sedan": "car", "suv": "car", "vehicle": "car", "automobile": "car", "taxi": "car", "jeep": "car",
    "lorry": "truck", "van": "truck",
    "sofa": "couch", "television": "tv", "monitor": "tv", "phone": "cell phone", "mobile": "cell phone",
    "plant": "potted plant", "ball": "sports ball", "table": "dining table", "bag": "handbag",
}

# Words that describe an object rather than name it ("orange car" is a car).
MODIFIERS = frozenset({
    "red", "blue", "green", "yellow", "black", "white", "orange", "pink", "purple", "brown",
    "grey", "gray", "dark", "light", "small", "big", "large", "little", "tiny", "huge",
})

_LEADING_ARTICLE = re.compile(r"^(?:the|a|an|that|this|those|these|all the|all|every)\s+")


def clean_phrase(text: str) -> str:
    """Lower-case, strip trailing punctuation / politeness and a leading article."""
    text = str(text or "").lower().strip()
    text = re.sub(r"[.?!,;:]+$", "", text)
    text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text)
    text = _LEADING_ARTICLE.sub("", text)
    return text.strip()


def _lookup(word: str) -> str | None:
    """COCO class for one word (exact, synonym or simple plural)."""
    candidates = (word, word[:-1]) if word.endswith("s") else (word,)
    for cand in candidates:
        if cand in COCO_SET:
            return cand
        if cand in ALIASES:
            return ALIASES[cand]
    return None


def to_coco_class(phrase: str) -> str:
    """Map a free-text object phrase to a COCO class name, or "none".

    The first noun that maps wins ("man walking near the car" -> person), two-word
    classes are matched first ("cell phone"), and a colour or size word is skipped
    when a mappable noun follows it ("orange car" -> car, "the orange" -> orange).
    """
    p = clean_phrase(phrase)
    if not p:
        return "none"
    if p in COCO_SET:
        return p
    if p in ALIASES:
        return ALIASES[p]

    words = re.findall(r"[a-z]+", p)
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt and f"{word} {nxt}" in COCO_SET:
            return f"{word} {nxt}"
        cls = _lookup(word)
        if cls is None:
            continue
        if word in MODIFIERS and nxt is not None and _lookup(nxt) is not None:
            continue
        return cls
    return "none"
