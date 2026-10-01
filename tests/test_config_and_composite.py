from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from editing.config import load_config
from editing.insert import CompositeInserter

ROOT = Path(__file__).resolve().parents[1]


def test_config_files_load():
    default = load_config(ROOT / "configs" / "default.yaml")
    assert default.parser.backend == "cascade" and default.inpaint.backend == "migan"
    core = load_config(ROOT / "configs" / "core_removal.yaml")
    assert core.parser.backend == "regex" and core.grounding.open_vocab is False
    assert core.insert.composite_fallback is False


def test_overrides_and_unknown_keys():
    cfg = load_config(overrides={"insert": {"steps": 6}, "threads": 2})
    assert cfg.insert.steps == 6 and cfg.threads == 2 and cfg.insert.size == 512
    with pytest.raises(ValueError):
        load_config(overrides={"insert": {"stepz": 6}})


def test_weight_path_resolution():
    cfg = load_config(overrides={"weights_dir": "w"})
    assert cfg.weight_path("migan_pipeline_v2.onnx") == Path("w") / "migan_pipeline_v2.onnx"


def _asset_dir(tmp_path: Path) -> Path:
    rgba = np.zeros((60, 40, 4), np.uint8)
    rgba[10:50, 10:30] = (200, 50, 50, 255)  # opaque red block on a transparent canvas
    Image.fromarray(rgba).save(tmp_path / "dog_bg.png")
    return tmp_path


def test_find_asset_by_head_noun_and_alias(tmp_path):
    inserter = CompositeInserter(_asset_dir(tmp_path))
    assert inserter.find_asset("golden retriever puppy").name == "dog_bg.png"
    assert inserter.find_asset("horse") is None


def test_composite_insert_paints_inside_box_only(tmp_path):
    inserter = CompositeInserter(_asset_dir(tmp_path))
    background = np.full((120, 160, 3), 128, np.uint8)
    out = inserter.insert(background, (60, 20, 100, 110), inserter.find_asset("dog"), shadow_strength=0)
    assert out.shape == background.shape
    assert (out[:, :50] == 128).all()  # left of the box untouched
    assert np.abs(out[60:100, 70:90].astype(int) - 128).max() > 20  # object painted
