# Solution approach

Text-driven object removal and replacement under CPU-only constraints.
Companion documents: [benchmark report](benchmark_report.md) (all measurements) and the
experiment notebooks in [`notebooks/experiments/`](../notebooks/experiments).

## 1. Problem and constraints

The user describes an edit in plain English: *"could you get rid of the dark sedan on the highway?"*,
*"turn the kitten into a golden retriever puppy"*. The system must remove the named object, or
replace it with a text-defined one, under four constraints:

| Constraint | Consequence for the design |
|---|---|
| CPU-only inference | Every model is benchmarked on a 2-vCPU Intel Xeon (2.20 GHz, 12.7 GB RAM). |
| Smallest practical models | Footprint is reported per component and per deployment profile. |
| Text-driven interface | The parser must handle free-form phrasing, "which one" and non-COCO objects. |
| Diffusion and non-diffusion explored | Both families are measured; the choice is made per stage. |

## 2. Design principle: decompose the edit

An end-to-end instruction-editing model (InstructPix2Pix class, around 1 B parameters) offers little
control and takes minutes per image on this CPU. The edit is split into five stages, each with the
smallest model that does its job, each timed and sized on its own:

```
instruction ─► 1 Parse ─► 2 Ground ─► 3 Segment ─► 4 Remove ─► 5 Insert (replace only)
               action,     object →     box →         mask →       target text →
               object,     box          pixel mask    background   new object
               which,
               target
```

Because each stage is measured separately, the bottleneck is always visible, and any stage can be
swapped without touching the others. The code mirrors this: `src/editing/{parse,ground,segment,inpaint,insert}.py`,
composed by `pipeline.Editor`.

## 3. Approaches explored

| Exp | Question | Approach | Family | Outcome |
|---|---|---|---|---|
| E0 | Cheapest baseline? | YOLOv8n box → OpenCV Telea | classical | 0.32 s, 6.3 MB, box-shaped smear |
| E1 | Does a tight mask help? | + MobileSAM mask | classical | mask yes, fill no; SAM costs 2–5 s |
| E2-A | Can diffusion erase? | SD-1.5 inpainting over the object | diffusion | re-creates the object; ~400 s, 4.1 GB |
| E3 | One-pass learned inpainting? | MI-GAN vs LaMa | learned, non-diffusion | MI-GAN: 1 s, 27 MB → removal |
| E4 | Free text, any object? | Qwen2.5-0.5B parser, YOLO-World, YOLOE | LLM + open vocabulary | grounding 3/10 → 8/10 |
| E5 | Text-defined replacement? | erase first, then SD / LCM / TAESD or composite | diffusion and compositing | 449 s → 98 s |
| E6 | What else can be cut? | YOLOv8n-seg, ONNX export, int8, thread and knob sweeps | optimisation | 2 kept, 3 rejected |

### Why diffusion failed at removal (E2-A)

SD-1.5 inpainting painted a shrub at 256 px and a new car at 512 px into the car-shaped hole. Three causes:

1. **Shape prior** – a car-shaped hole on a road invites the model to paint a car.
2. **Latent leak** – with `strength` 0.85, about 15 % of the original latent survives, so the object comes back.
3. **No negation** – CLIP text conditioning reads "no object" as "object".

A Telea pre-fill, a box-shaped mask and a negative prompt fixed the output but left ~400 s and 4.1 GB
for a task a 27 MB GAN does in one second. Removal therefore never uses diffusion.

### Why MI-GAN (E3)

MI-GAN (Picsart, Places2, 512 px) ships as a 26.8 MB ONNX graph that crops around the hole,
inpaints in one forward pass and blends back. LaMa (big-lama) has an image-wide receptive field
through Fourier convolutions and gives cleaner texture on large holes, but it is 7× larger
(196 MB) and 4–17× slower to inpaint (its cost grows with the hole). MI-GAN is the default;
LaMa is kept as a quality reference (`inpaint.backend: lama`).

## 4. Final architecture

```
instruction
   │
   ▼
1 PARSE    regex rules: verb patterns, instance words, alias table               0 MB
           unusual phrasing only: Qwen2.5-0.5B-Instruct Q4_K_M on llama.cpp      469 MB
           (JSON schema → grammar, loaded on first use)
   │        → action, object phrase, COCO class, which, target, target prompt
   ▼
2 GROUND   COCO class?  ── yes ─► YOLOv8n-seg: box + mask in one pass            6.7 MB
              │ no / nothing found
              └──────────────────► YOLO-World-S on the free-text phrase           25 MB + 338 MB CLIP text encoder
           select instance: best | left | right | top | bottom | center | largest | smallest | all
   ▼
3 SEGMENT  seg mask from step 2, or MobileSAM on the open-vocabulary path         38.8 MB
           removal mask = dilated mask ∪ box, extended over the contact shadow
   ▼
4 REMOVE   MI-GAN 512, ONNX Runtime                                               26.8 MB
   ▼
5 INSERT   (replace only) on the clean background, inside a box-shaped placement mask
           diffusion: SD-1.5 inpainting UNet + text encoder, LCM-LoRA fused,
                      4 steps, guidance 1, TAESD tiny VAE, 512 px square crop      3.76 GB
           composite: stored cutout + LAB colour match + contact shadow            0.4 MB
```

### Model choices

| Stage | Model | Weights | Params | Why this one |
|---|---|---|---|---|
| Parse | regex rules + alias table | 0 MB | — | common phrasing, "which one" and synonyms in < 1 ms |
| Parse (unusual) | Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (llama.cpp) | 469 MB | 0.49 B | smallest instruct model that handles free-form phrasing; grammar-constrained JSON |
| Ground | YOLOv8n-seg | 6.7 MB | 3.4 M | COCO detection and instance masks in one 0.2–0.9 s pass |
| Ground (fallback) | YOLO-World-S v2 + CLIP ViT-B/32 text encoder | 25 + 338 MB | — | any noun, no retraining; 8/10 vs 3/10 grounding |
| Segment (fallback) | MobileSAM (TinyViT) | 38.8 MB | 10.1 M | masks for open-vocabulary boxes |
| Remove | MI-GAN 512 (ONNX) | 26.8 MB | — | one pass, ~1 s, good on roads, grass, sky |
| Insert | SD-1.5 inpainting UNet + text encoder, LCM-LoRA fused | 3.75 GB (fp32) | 983 M | the only option that draws an arbitrary text-defined object |
| Insert | TAESD | 9.3 MB | 2.4 M | replaces the 84 M-param VAE: −62 s per replacement |

### Key decisions and the evidence for each

1. **Erase first, then insert (E5).** Generating over the old object leaked it back (E2-A). Erasing
   with MI-GAN first gives the generator a clean background, and the placement mask is a box sized
   for the new object instead of the old silhouette.
2. **Generous, box-like masks.** The removal mask is the object mask dilated by 15 iterations of a
   3×3 kernel, OR its box, extended downward by 15 % of the box height to cover the contact shadow.
   Inpainters smear any object or shadow pixel left at the border. This is also why the coarser
   YOLOv8n-seg mask is good enough (mask IoU 0.82–0.95 against MobileSAM, same visual result).
3. **LCM-LoRA plus TAESD (E5, E6).** LCM-LoRA cuts the UNet from 10 steps × 2 passes (CFG) to
   4 steps × 1 pass, 5× fewer UNet evaluations, but replacement got only 2.7× faster: the full VAE
   (encode and decode at 512 px) became the bottleneck. TAESD removed 62 s per image. The sweep
   picked 4 steps at 512 px: 2 steps ghost the object; 384 px saves 43 % but loses detail.
4. **Fast path first, open vocabulary on demand.** COCO classes never load YOLO-World, the CLIP text
   encoder or MobileSAM. In the final run the fallback also rescued two replacements whose LLM
   class label found nothing on the fast path.
5. **Alias mapping for the detector class.** Forcing the 0.5B LLM to choose from an 80-name enum made
   constrained decoding drift to the nearest legal prefix (kitten → kite, SUV → sports ball,
   cyclist → carrot). The repository maps the LLM's free-text object through a deterministic alias
   table instead (`parser.class_mapping: alias`); re-scoring the saved LLM outputs gives 26/26
   classes correct, up from 19/26. The benchmarked runs used the enum label; latency is unchanged
   because the prompt is the same.
6. **Lazy loading and deployment profiles.** Components load on first use, so each profile pays only
   for what it needs (section 6).
7. **Rules first, LLM on demand.** The LLM was 72 % of removal time, and most of its gain over the E4
   regex baseline came from instance words ("on the left", "biggest") and trimming the object phrase,
   which rules can do. The default parser (`parser.backend: cascade`) runs extended rules first and
   loads the LLM only when they cannot parse an instruction or flag it as unusual: a vague or long
   object phrase, an ambiguous target, or a negation. The rules score 26/26 on the parser test set,
   but they were written against it, so that is a regression check, not a held-out score.

## 5. Optimisations tried and rejected

| Idea | Result | Decision |
|---|---|---|
| ONNX Runtime export of YOLOv8n-seg | 217 ms vs 173 ms in PyTorch; 13.2 MB fp32 vs 6.7 MB fp16; identical masks (IoU 0.999) | rejected; OpenVINO is the next candidate on Intel CPUs |
| int8 dynamic quantisation of UNet Linear layers | −5 % time, UNet 3.28 → 2.51 GB, visibly ghosted output (CLIP p 0.994 → 0.981) | rejected; convolutions dominate and stay fp32 |
| More CPU threads | MobileSAM 5.1 → 2.8 s in one session, slower in another; YOLOv8n slower with 2 threads | not a reliable win on 2 shared vCPUs |
| MobileSAM on the fast path | 3.1–3.7 s vs 0.2–0.9 s for YOLOv8n-seg, +39 MB, same removal quality | replaced by YOLOv8n-seg |
| LaMa as the default remover | cleaner on large holes, 7× larger, 4–17× slower | kept as an option |

## 6. Trade-offs: quality, footprint, latency

| Deployment profile | Config | Weights | CPU latency (2 vCPU) | What it buys |
|---|---|---|---|---|
| 1 · remove COCO objects | `configs/core_removal.yaml` | 33.5 MB | ≈1.8 s | common phrasing, 80 classes |
| 2 · + unusual phrasing | `parser.backend: cascade` | 502 MB | ≈10.5 s when the LLM runs | free-form phrasing the rules cannot read |
| 3 · + any noun | `grounding.open_vocab: true` | 903 MB | ≈12.8 s | non-COCO objects (ladder) |
| 4 · + text-defined replacement | `configs/default.yaml` | 4.66 GB | ≈88 s | any replacement described in words |
| 4b · fast replacement | `insert.mode: composite` | 33.9 MB | ≈3 s | real objects from a small library |

Removal stays cheap at every profile; natural language and open vocabulary buy flexibility, not
pixel quality; text-defined replacement is the only step that needs gigabytes, and inside that band
the work moved it from 449 s to 88 s.

## 7. Limitations and next steps

* **Replacement latency** – the UNet is 83 % of replacement time. Next: OpenVINO or ONNX with
  weight-only int8, or a distilled UNet (BK-SDM, 1–2-step SD-Turbo).
* **Parser latency** – 72 % of removal time when the LLM runs, mostly re-reading a long system prompt;
  the cascade now skips it for common phrasing. Next: drop the class list from the prompt (alias
  mapping makes it unnecessary) and cache the prefix.
* **Large holes next to other objects** – MI-GAN copies neighbouring texture. Next: route by mask
  area, MI-GAN for small holes and LaMa for large ones.
* **Object relations** – removing a bike leaves its rider. Next: remove attached instances together.
* **Domain gap** – no detector found a cartoon hat on a sticker sheet.
* **Low-confidence fallback** – a 0.09-confidence box removed a bottle instead of a candle; a higher
  floor or asking the user is safer.
* **Quality metric** – CLIP zero-shot p(target) saturates at 0.97–0.99. Next: a paired synthetic set
  (objects pasted onto clean backgrounds) for PSNR and LPIPS.
* **Halo around generated objects** – from the box-shaped placement mask; a soft elliptical mask is the fix.
