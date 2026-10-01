"""Text-driven object removal and replacement on CPU."""

from editing.config import EditorConfig, load_config
from editing.pipeline import Editor
from editing.types import Detection, EditResult, ObjectNotFoundError, ParsedInstruction

__version__ = "1.0.0"

__all__ = [
    "Detection",
    "EditResult",
    "Editor",
    "EditorConfig",
    "ObjectNotFoundError",
    "ParsedInstruction",
    "load_config",
]
