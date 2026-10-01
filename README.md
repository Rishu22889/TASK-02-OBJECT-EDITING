# Text-driven object removal and replacement on CPU

Remove or replace an object in a photo from one plain-English sentence, using CPU-only inference
and the smallest practical models.

```
"Could you get rid of the dark sedan on the highway?"   → removed in 1.8 s with 33.5 MB of weights
"swap the car for a red bicycle"                        → replaced in ~80–90 s (SD-1.5 + LCM + TAESD)
```

| Original | Remove | Replace |
|---|---|---|
| ![](data/samples/car_2.jpg) | ![](results/E6_regex_session/final_car_2_could_you_get_rid_of_the_dark_sedan_on_t.jpg) | ![](results/E6_regex_session/final_car_2_swap_the_car_for_a_red_bicycle.jpg) |

## Results at a glance

Measured on a 2-vCPU Intel Xeon @ 2.20 GHz (Colab CPU runtime). Details in the
[benchmark report](docs/benchmark_report.md).

| Deployment profile | Weights | CPU latency / image |
|---|---|---|
| Remove COCO objects (regex parser) | **33.5 MB** | **≈1.8 s** |
| + LLM for unusual phrasing (Qwen2.5-0.5B, 4-bit) | 502 MB | ≈10.5 s when the LLM runs |
| + any noun (YOLO-World open vocabulary) | 903 MB | ≈12.8 s |
| + text-defined replacement (SD-1.5 inpaint + LCM-LoRA + TAESD) | 4.66 GB | ≈88 s (baseline: 449 s) |
| Fast replacement from an asset library (no diffusion) | 33.9 MB | ≈3 s |

## How it works

The edit is decomposed into five stages, each with the smallest model that does its job:

```
instruction ─► Parse ─────────► Ground ─────────────► Segment ─────────► Remove ─► Insert (replace only)
               regex rules;     YOLOv8n-seg (COCO)     same-pass mask     MI-GAN    SD-1.5 inpaint + LCM-LoRA
               Qwen2.5-0.5B     YOLO-World fallback    MobileSAM for       (ONNX)    + TAESD, or a stored cutout
               Q4 if unusual    for any other noun     open-vocab boxes
```

Removal never uses diffusion (it re-creates the object and costs ~400 s); replacement erases first and
then generates on the clean background. Why each model was chosen, and what was tried and rejected, is
in the [solution approach](docs/approach.md).

## Setup

Python 3.10+ (benchmarked on 3.13). CPU only.

```bash
pip install -r requirements.txt               # core: detection, segmentation, removal, composite
pip install -r requirements-llm.txt           # free-form English (llama.cpp CPU wheel)
pip install -r requirements-diffusion.txt     # text-defined replacement
pip install -e .                              # makes `editing` importable / the `edit-image` command

python scripts/download_weights.py --profile full    # or: core | nl | open-vocab
```

Weights go to `weights/` (not committed). Profiles: `core` ≈ 34 MB, `nl` ≈ 0.5 GB,
`open-vocab` ≈ 0.9 GB, `full` ≈ 4.7 GB. Anything missing is also downloaded on first use.

On Colab: open [`notebooks/demo.ipynb`](notebooks/demo.ipynb), which runs the steps above.

## Usage

Command line:

```bash
python -m editing data/samples/car_2.jpg "Could you get rid of the dark sedan on the highway?" -o outputs/removed.jpg
python -m editing data/samples/cat.jpg "turn the kitten into a golden retriever puppy" -o outputs/puppy.jpg
python -m editing data/samples/car_2.jpg "replace the car with a bicycle" --mode composite -o outputs/fast.jpg
python -m editing data/samples/car_2.jpg "remove the car" -c configs/core_removal.yaml   # 33.5 MB profile
```

The command prints the parsed instruction, the grounding route, per-stage timings and the weights loaded.
`parsed.parser` shows who read the instruction: the regex rules handle common phrasing in under 1 ms, and
the LLM loads only for unusual phrasing (`--parser regex` never loads it).

Python:

```python
from editing import Editor, load_config
from editing.utils import read_image, write_image

editor = Editor(load_config("configs/default.yaml"))
result = editor.edit(read_image("data/samples/park_2.jpg"), "I don't want the man walking in this photo")
write_image("outputs/park_2.jpg", result.image)
print(result.parsed, result.route, result.timings)   # timings in seconds per stage
```

Configuration lives in [`configs/default.yaml`](configs/default.yaml) (final pipeline) and
[`configs/core_removal.yaml`](configs/core_removal.yaml) (smallest profile).

## Reproducing the benchmarks

```bash
python scripts/run_benchmark.py --set final                                   # final pipeline, 8 cases
python scripts/run_benchmark.py --config configs/core_removal.yaml --set core --repeats 3
python scripts/evaluate_parser.py --llm                                       # regex vs LLM vs cascade, 26 instructions
python -m pytest                                                              # unit tests (no models needed)
```

Each experiment E0–E6 is a self-contained Colab notebook in
[`notebooks/experiments/`](notebooks/experiments); their CSV outputs are in
[`benchmarks/`](benchmarks) and their result images in [`results/`](results).

## Repository layout

```
configs/                 pipeline configs (final, smallest profile)
src/editing/
  parse.py               regex rules first, LLM (llama.cpp + JSON grammar) for unusual phrasing
  vocabulary.py          COCO names, synonyms, phrase → class mapping
  ground.py              YOLOv8n-seg, YOLO-World, instance selection (left / largest / all …)
  segment.py             MobileSAM, removal mask, placement mask
  inpaint.py             MI-GAN (ONNX), LaMa
  insert.py              SD-1.5 inpaint + LCM-LoRA + TAESD, composite fast mode
  pipeline.py            Editor: lazy-loaded components, routing, per-stage timing
  cli.py, config.py, types.py, utils.py, weights.py
scripts/                 download_weights, run_benchmark, evaluate_parser
notebooks/demo.ipynb     working notebook: removal and replacement on sample images
notebooks/experiments/   E0 … E6 experiment notebooks with outputs
data/                    sample images, parser test set, benchmark cases
assets/                  cutouts for the composite fast mode
benchmarks/              CSV / JSON results per experiment
results/                 before / after images per experiment
docs/                    approach.md, benchmark_report.md
tests/                   unit tests
```

## Documentation

* [Solution approach](docs/approach.md): explored approaches, final architecture, model choices,
  optimisation decisions, trade-offs.
* [Benchmark report](docs/benchmark_report.md): CPU latency, model size and parameters, dependency
  footprint, output quality, bottlenecks, failure cases.

## Third-party models

YOLOv8n-seg and YOLO-World (Ultralytics), MobileSAM, MI-GAN (Picsart AI Research), LaMa,
Qwen2.5-0.5B-Instruct (Qwen), Stable Diffusion 1.5 inpainting, LCM-LoRA (latent-consistency),
TAESD and CLIP. Each is downloaded from its official source and used under its own licence.
