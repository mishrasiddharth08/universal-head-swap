# Z-Image head swap

## Setup

1. Load a Z-Image checkpoint with its matching text encoder and VAE in Forge.
2. Put the target picture in plain **img2img**. Enable **Universal Head Swap**.
3. In **Swap**, select a Z-Image **Character LoRA** and its training trigger. Set a positive LoRA strength. BFS is not used.
4. Open **Z-Image · character LoRA inpainting**. Choose **Turbo** or **Base** to match the checkpoint. Compare identity change **0.65–0.75** with character strength **0.8**; increasing either can damage likeness or proportions. These are tested starting points, not guarantees.
5. Keep **Head only**. Use **Preview setup** to inspect the edit mask, then Forge **Generate**.

Turbo uses Forge CFG 1, so negative guidance is inactive. Forge retains your sampler and step count; around 8–9 steps is a starting point for Turbo, while Base needs its normal settings. Character likeness depends on the trained adapter and its strength. Headshots are optional comparison references, not image conditioning for Z-Image.

The face detector provides the default mask. A white-on-black custom mask takes priority. Crops are fitted without stretching, sampled with Forge's native inpainting mask, then blended into the original-size picture. Existing geometry checks, detail controls, cleanup prompts and output reports remain available. Full-image mode and no-reference diagnostic mode are rejected for this route to avoid unintended whole-scene edits. Reference-only latent sharpening has no effect on Z-Image.

## Spot cleanup and optional SAM3

In **Appearance → Remove small marks / jewelry**, upload the same target photo and paint small tattoo strokes, studs or symbols white. CPU spot repair modifies only painted pixels. Clear the editor before changing targets. Broad masks above 3% of the image are rejected; large tattoos and objects crossing body edges need manual retouching.

### Get SAM3 working — simple step-by-step

Official weights live at [facebook/sam3](https://huggingface.co/facebook/sam3/tree/main). Any **one** of these routes installs everything:

**Route A — inside Forge (recommended)**
1. Open **Tab 2 · Appearance → Auto detect · optional SAM3**.
2. If your Hugging Face account already has access to `facebook/sam3`, just click **Download / set up SAM3 once**. If this machine is not logged in, paste a read token in the optional token field first ([request access](https://huggingface.co/facebook/sam3) if you have not been approved yet).
3. Wait for the status line to confirm the checkpoint and runtime. The weights are saved to `Forge/models/sam3/sam3.pt`; the official source is pinned and installed dependency-free into `Forge/models/sam3/runtime` — Forge packages are never upgraded.
4. Restart Forge. Done — SAM3 is picked up automatically and is the default Z-Image mask source.

**Route B — manual download (no login needed)**
1. Download `sam3.pt` from an ungated community mirror of the official checkpoint (search Hugging Face for `sam3.pt`), or copy an existing official `sam3.pt` (~3.4 GB).
2. Place it at `Forge/models/sam3/sam3.pt` (a `models/SAM 3/` folder is also scanned automatically).
3. In Forge, expand the same accordion and click **Download / set up SAM3 once** once — this installs only the runtime code (no gated download happens).
4. Restart Forge.

**Route C — already have a `.safetensors` export**
Place it anywhere and point **Local SAM3 checkpoint** at it. The extension loads it through a safetensors fallback; if its key layout does not match Meta's model, it reports the mismatch instead of guessing. Routes A and B use the official checkpoint layout; runtime and hardware compatibility still need verification.

**After setup**
- Z-Image masks: **Z-Image mask → SAM3 (recommended)** (default). The fast face detector takes over automatically on any photo where SAM3 fails, when available; inspect fallback messages and masks.
- Speed on modern GPUs: set **SAM3 device → GPU (CUDA)** and **precision → BFloat16**. Available precision modes depend on the installed runtime, PyTorch and hardware. INT modes require `torchao`; not every mode has been GPU-tested.
- Spot detection (tattoos, piercings, symbols) uses the same installation; nothing else to configure.

Nothing downloads during generation. Missing checkpoints or runtime produce a clear message plus face-detector fallback when a supported fallback is available.

Choose Tattoos, Body piercings, Cross symbols, or Other religious symbols. Add specific visible object names when useful. Click **Detect candidates**, review numbered boxes, select items, then **Use selected items**. Nothing is selected automatically. Correct the paint and use **Preview removal** before Generate. Detection is not exhaustive; it can miss or mislabel objects. These are object proposals, not an inference about anyone's religion.

Detection runs at a maximum 1024-pixel side and returns at most 24 proposals. SAM3 unloads after scanning. For optional SAM3 head masking, choose SAM3 under Swap; the checkpoint is auto-found in `models/sam3` or `models/SAM 3`.

## Inpainting repair and identity limits

The model samples a solid mask slightly wider than the final blend. Paste-back uses the original soft mask, preserving mask-zero pixels. Partial denoise uses the original image latent; full denoise may use random latent. Random latent with partial denoise was rejected after live tests produced patterned faces.

Use a **Base character LoRA with Base**, or a **Turbo character LoRA with Turbo**. LoRA subfolders are scanned through Forge. Uploaded headshots measure optional likeness only; they do not condition Z-Image. Select the adapter and its trained trigger in **Swap**, rather than the BFS field. Test with Spectrum disabled first; enable it only after comparing the same seed and settings.

## Memory and compatibility

The extension uses Forge's existing Z-Image loader, LoRA handling, sampler and offload policy. It creates no second diffusion model and performs no reference VAE encodes. The crop respects the selected generation size and available-memory estimate. Quantization support depends on the installed Forge loader; every quantization and physical 6/8 GB GPU has not been tested.

## Measured results

Five Base and five Turbo outputs completed with the matching character LoRAs above. All ten preserved mask-zero pixels. Eight passed independent proportion checks; two turned-head cases still need review. See [validation](../VALIDATION.md) for settings, timings, likeness scores and hardware limits.

## Sources

- [Community workflow discussion](https://www.reddit.com/r/StableDiffusion/comments/1qz9lzb/simple_effective_and_fast_zimage_headswap_for/)
- [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- [Meta SAM3](https://github.com/facebookresearch/sam3)
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image)

This is an independent Forge implementation of character-LoRA inpainting. It does not import the linked ComfyUI workflow or claim identical output. See [validation](../VALIDATION.md) for measured results.
