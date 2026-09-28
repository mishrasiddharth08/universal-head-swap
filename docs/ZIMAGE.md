# Z-Image head swap

## Setup

1. Load a Z-Image checkpoint with its matching text encoder and VAE in Forge.
2. Put the target picture in plain **img2img**. Enable **Universal Head Swap**.
3. In **Swap**, select a Z-Image **Character LoRA** and its training trigger. Set a positive LoRA strength. BFS is not used.
4. Open **Z-Image · character LoRA inpainting**. Choose **Turbo** or **Base** to match the checkpoint. Start with identity change **0.65**; increasing it changes more of the masked face.
5. Keep **Head only**. Use **Preview setup** to inspect the edit mask, then Forge **Generate**.

Turbo uses Forge CFG 1, so negative guidance is inactive. Forge retains your sampler and step count; around 8–9 steps is a starting point for Turbo, while Base needs its normal settings. Character likeness depends on the trained adapter and its strength. Headshots are optional comparison references, not image conditioning for Z-Image.

The face detector provides the default mask. A white-on-black custom mask takes priority. Crops are fitted without stretching, sampled with Forge's native inpainting mask, then blended into the original-size picture. Existing geometry checks, detail controls, cleanup prompts and output reports remain available. Full-image mode and no-reference diagnostic mode are rejected for this route to avoid unintended whole-scene edits. Reference-only latent sharpening has no effect on Z-Image.

## Spot cleanup and optional SAM3

In **Appearance → Remove small marks / jewelry**, upload the same target photo and paint small tattoo strokes, studs or symbols white. CPU spot repair modifies only painted pixels. Clear the editor before changing targets. Broad masks above 3% of the image are rejected; large tattoos and objects crossing silhouettes need manual retouching.

For automatic proposals, expand **Auto detect → Download / set up SAM3 once**. The official gated model requires approved Hugging Face access. Use an existing local login or the optional read-token field; the extension does not save that token or accept licenses for you.

The checkpoint is saved to **Forge/models/sam3/sam3.pt** and reused. Optional official SAM3 source is pinned and installed without dependencies into **Forge/models/sam3/runtime**; Forge packages are not upgraded. Generation never triggers downloads. Missing runtime dependencies produce an error rather than silently altering Forge.

Choose Tattoos, Body piercings, Cross symbols, or Other religious symbols. Add specific visible object names when useful. Click **Detect candidates**, review numbered boxes, select items, then **Use selected items**. Nothing is selected automatically. Correct the paint and use **Preview removal** before Generate. Detection is not exhaustive; it can miss or mislabel objects. These are object proposals, not an inference about anyone's religion.

Detection runs on CPU at a maximum 1024-pixel side and returns at most 24 proposals. SAM3 unloads after scanning. The default head mask still uses the face detector. For optional SAM3 head masking, choose SAM3 under Swap and enter the checkpoint path above. Real SAM3 detection remains unvalidated on this host until setup completes.

## Memory and compatibility

The extension uses Forge's existing Z-Image loader, LoRA handling, sampler and offload policy. It creates no second diffusion model and performs no reference VAE encodes. The crop respects the selected generation size and available-memory estimate. Quantization support depends on the installed Forge loader; every quantization and physical 6/8 GB GPU has not been tested.

## Sources

- [Community workflow discussion](https://www.reddit.com/r/StableDiffusion/comments/1qz9lzb/simple_effective_and_fast_zimage_headswap_for/)
- [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- [Meta SAM3](https://github.com/facebookresearch/sam3)
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image)

This is an independent Forge implementation of character-LoRA inpainting. It does not import the linked ComfyUI workflow or claim identical output. See [validation](../VALIDATION.md) for measured results.
