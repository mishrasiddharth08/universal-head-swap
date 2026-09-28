# Z-Image head swap

## Setup

1. Load a Z-Image checkpoint with its matching text encoder and VAE in Forge.
2. Put the target picture in plain **img2img**. Enable **Universal Head Swap**.
3. In **Swap**, select a Z-Image **Character LoRA** and its training trigger. Set a positive LoRA strength. BFS is not used.
4. Open **Z-Image · character LoRA inpainting**. Choose **Turbo** or **Base** to match the checkpoint. Start with identity change **0.65**; increasing it changes more of the masked face.
5. Keep **Head only**. Use **Preview setup** to inspect the edit mask, then Forge **Generate**.

Turbo uses Forge CFG 1, so negative guidance is inactive. Forge retains your sampler and step count; around 8–9 steps is a starting point for Turbo, while Base needs its normal settings. Character likeness depends on the trained adapter and its strength. Headshots are optional comparison references, not image conditioning for Z-Image.

The face detector provides the default mask. A white-on-black custom mask takes priority. Crops are fitted without stretching, sampled with Forge's native inpainting mask, then blended into the original-size picture. Existing geometry checks, detail controls, cleanup prompts and output reports remain available. Full-image mode and no-reference diagnostic mode are rejected for this route to avoid unintended whole-scene edits. Reference-only latent sharpening has no effect on Z-Image.

## Optional SAM3

SAM3 is not required. If its official Python package and checkpoint are already installed, choose **SAM3 (optional)** and enter the full local checkpoint path. No package or model is downloaded automatically. Use a trusted upstream checkpoint.

This integration calls Meta's image processor directly, not ComfyUI nodes. Segmentation runs on CPU, uses a maximum 1024-pixel input side, caches at most 8 MiB of CPU masks, and releases the segmentation model before diffusion. It selects the detection overlapping the chosen face and restricts automatic masks to that head region. Missing dependencies, missing weights or failed selection stop with a clear message; select the fast detector or supply a custom mask instead.

SAM3 text-mask quality is not guaranteed. [ComfyUI-SAM3 issue #98](https://github.com/PozzettiAndrea/ComfyUI-SAM3/issues/98) reports a text-segmentation regression. Preview every automatic mask. Real SAM3 model execution has not been validated in this Forge environment.

## Memory and compatibility

The extension uses Forge's existing Z-Image loader, LoRA handling, sampler and offload policy. It creates no second diffusion model and performs no reference VAE encodes. The crop respects the selected generation size and available-memory estimate. Quantization support depends on the installed Forge loader; every quantization and physical 6/8 GB GPU has not been tested.

## Sources

- [Community workflow discussion](https://www.reddit.com/r/StableDiffusion/comments/1qz9lzb/simple_effective_and_fast_zimage_headswap_for/)
- [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- [Meta SAM3](https://github.com/facebookresearch/sam3)
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image)

This is an independent Forge implementation of character-LoRA inpainting. It does not import the linked ComfyUI workflow or claim identical output. See [validation](../VALIDATION.md) for measured results.
