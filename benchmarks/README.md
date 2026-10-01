# Benchmark data

All runs: Colab CPU runtime, Intel Xeon @ 2.20 GHz, 2 vCPU, 12.7 GB RAM (see `*system_info.json`).
Summaries and discussion: [`docs/benchmark_report.md`](../docs/benchmark_report.md).

| File | Experiment | Content |
|---|---|---|
| `e0_benchmark.csv`, `e0_results.csv` | E0 | YOLOv8n box + Telea removal, box paste replacement |
| `e1_benchmark.csv` | E1 | + MobileSAM mask |
| `e2a_benchmark.csv` | E2-A | SD-1.5 inpainting for removal and replacement (512 px, 10 steps) |
| `e3_migan_benchmark.csv`, `e3_lama_benchmark.csv` | E3 | learned inpainting, per stage, 3 runs per image |
| `e4_parser_eval.csv` | E4 | regex vs LLM on 26 instructions, per field |
| `e4_grounding_eval.csv` | E4 | 4 grounding systems on 10 instructions |
| `e4_benchmark.csv`, `e4_model_sizes.csv` | E4 | end-to-end removal latency, component sizes |
| `e5_replace_benchmark.csv`, `e5_replace_summary.csv` | E5 | 4 replacement variants, per stage, CLIP scores |
| `e6_regex_session/` | E6, session 1 | threads, segmentation ablation, ONNX export, diffusion sweep, final pipeline with the regex parser, dependency footprint |
| `e6_llm_session/` | E6, session 2 | the same ablations, final pipeline with the LLM parser, int8 UNet |
| `summary_all_experiments.csv` | all | latency and weights per experiment (session 1) |
| `repo_check/` | repository | core profile run with `scripts/run_benchmark.py` on an 8-core arm64 laptop |

Notes

* `e6_llm_session/` was extracted from the printed outputs of
  `notebooks/experiments/E6_final_pipeline_llm_session.ipynb` (its CSVs were written over on the run
  machine). `final_pipeline_sizes_as_reported.csv` shows the notebook's own size scan, which missed the
  cached 337.6 MB CLIP text encoder; the report adds it back.
* `summary_all_experiments.csv` predates session 2: its E2-A row comes from `e2a_benchmark.csv`, and
  its final-pipeline rows are from session 1 (regex parser).
* Shared vCPUs made identical code up to 2× slower in session 2; compare numbers within one session.
