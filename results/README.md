# Result images

| Folder | Experiment | Contents |
|---|---|---|
| `E0/` | YOLOv8n box + Telea | input / mask / output strips for removal; box paste for replacement |
| `E1/` | + MobileSAM mask | removal (`remove_1.png`) and paste replacement (`replace1.png`) |
| `E2/` | SD-1.5 inpainting | removal with pre-fill + box mask + negative prompt (`remove_*.png`), replacement (`replace_1.png`) |
| `E3/` | MI-GAN vs LaMa | original / mask / MI-GAN / LaMa strips |
| `E4/` | LLM parser + open vocabulary | `e4_grounding_grid.jpg` (4 systems × 10 instructions), `e4_removal_grid.jpg`, per-case outputs (`*_world_*` = YOLO-World, `*_yoloe_*` = YOLOE) |
| `E5/` | erase first, then insert | `e5_replacement_grid.jpg` (input, MI-GAN erase, variants A–D), per-variant outputs |
| `E6_regex_session/` | final pipeline, regex parser | thread scaling, segmentation ablation, diffusion sweep, final outputs, size-vs-latency plot |
| `E6_llm_session/` | final pipeline, LLM parser | the same figures for session 2, plus `int8_unet.png` |
