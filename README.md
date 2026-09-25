# Universal Head Swap for Forge Neo

**BFS head swapping in Forge's familiar workflow. One extension for FLUX.2 Klein 4B/9B and Qwen Image Edit 2.1 — the loaded checkpoint decides the mode.**

[Installation](#beginner-installation) · [Model files](#model-files) · [Step-by-step usage](#step-by-step-usage) · [UI tour](#ui-tour) · [Latest update](#latest-update) · [Testing status](#features-and-testing-status)

## Latest update

- **Dual-model engine:** Klein and Qwen Image Edit 2.1 in one extension; auto-switching with the checkpoint dropdown.
- **Full feature set on both models:** protected head edit, removal policies, presets, fine-grid cleanup and quality gates.
- **Family-locked adapters:** Klein and Qwen BFS/character LoRAs never cross-load; auto-match works in both modes.
- **Face-match auditing removed:** verify identity visually at 100% zoom.
- **Cleaner panel:** dedicated **Adapters** and **Memory** tabs in More options.

Full details in the [dated update history](CHANGELOG.md). 118 offline tests pass — restart Forge after updating.

## The idea

Use Forge's normal img2img tab, prompt, seed and **Generate** button.
Special controls stay inside a compact, collapsed **Universal Head Swap** panel.
The original picture is the body/scene reference; the selected headshot is the identity reference.
This is an independent community extension, not an official product. Inspect every output at 100% zoom.

## Features and testing status

- **FLUX.2 Klein 4B/9B:** inherited from the validated v6 codebase; reference latents, size-matched BFS adapters.
- **Qwen Image Edit 2.1:** new in this release — pixel-space references, vision-token prompts, Picture numbering remapped. Live GPU validation is still pending.
- **Organized controls:** everyday panel first; Adapters, Memory, Detail and Presets grouped in tabs.
- **Memory:** device-aware probing (never assumes GPU 0), bounded reference budgets, one OOM retry, 256 MiB encode cache.
- **Quantization:** all modern formats through Forge's own loaders; LoRAs stay on the runtime side path, never merged into quantized weights.
- **No Forge core edits:** delete the extension folder and Forge is 100% stock.

Automated tests do not prove every GPU, model file or Forge version works.
AMD/ROCm is supported in code but **not hardware-validated here**. DirectML/Vulkan are not implemented.
No promise is made for every VRAM size.

## Beginner installation

1. Stop Forge completely.
2. Choose **Code → Download ZIP** on this repository.
3. Extract the folder and place it inside `sd-webui-forge-classic/extensions/`.
4. Avoid double nesting. The correct path ends with:
   `extensions/universal-head-swap/scripts/klein_face_reference.py`.
5. Keep only one copy of the old `klein-head-swap` extension — disable or remove it first.
6. Start Forge normally. The extension installs no shared packages.
7. Refresh your browser with **Ctrl+F5** after updates.

## Model files

**Weights are not included.** Use your own downloaded checkpoints and LoRAs in Forge's normal folders.

| Component | Forge folder |
|---|---|
| FLUX.2 Klein 4B or 9B checkpoint, **or** Qwen Image Edit 2.1 checkpoint | `models/Stable-diffusion` |
| Matching BFS face/head-swap LoRA for the loaded family | `models/Lora` |
| Optional character LoRA (same family as the checkpoint) | `models/Lora` |

A Klein BFS LoRA is **not** interchangeable with Qwen — the extension enforces the family match and explains mismatches.

## UI tour

Everything lives inside Forge's normal img2img view. The extension adds a single collapsed **Universal Head Swap** accordion:

![Universal Head Swap panel layout](docs/img/ui-panel.svg)

Inside the panel:

- **Main accordion** — editing scope (Whole picture / Head only), the headshot gallery (up to 20), **Check setup** and the setup preview.
- **Look tab** — removal policies for tattoos, piercings, jewelry and forehead marks; guidance mode; hair, expression and style choices.
- **References tab** — selection mode, manual slot override, target face, framing, scoring and seed lock.
- **Adapters tab** — face-swap adapter with auto-match, character adapter and trigger, strict compatibility check, refresh list.
- **Memory tab** — maximum reference side, combined megapixel budget, encode caching and small-head detail boost.
- **Detail tab** — canvas/proportion checks, quality gates, fine-grid cleanup, head mask and finishing effects.
- **Presets and info tab** — save/load setups, prompt preview and the last generation report.

## Step-by-step usage

![Five-step workflow](docs/img/workflow.svg)

1. **Load** — select a Klein 4B/9B or Qwen Image Edit 2.1 checkpoint and its matching BFS LoRA.
2. **Target** — open **img2img** and upload the body/scene picture.
3. **Headshots** — enable **Universal Head Swap** and upload 1–20 sharp identity headshots. Prefer similar pose and lighting.
4. **Check setup** — one click shows the selected reference, edit area and fully resolved prompt before any generation.
5. **Generate** — press Forge's normal **Generate** button, then inspect **Show last generation report** and the output at 100% zoom.

Defaults just work: adapter on **Auto (match model)**, denoising at the BFS start value (1.0), fixed seed.

## First head swap

1. Select the checkpoint in Forge's normal dropdown.
2. Open **img2img**, upload the target, enable the accordion, add headshots.
3. Click **Check setup** and read the setup status.
4. Press **Generate**.
5. Check hair edges, forehead, ears, neck, shoulders and all visible skin at 100%.

## Reports and troubleshooting

- **Show last generation report** lists the selected reference slot, resolved prompt, encode counts and quality measurements.
- **Protected head edit** dedicates the sampling canvas to the head region and composites it into the untouched original; upload an optional mask inside the extension.
- **Keep original canvas** returns the original dimensions and aspect ratio.
- Stopping during reference preparation retains completed outputs and restores owned state.
- Reference encoding retries once at a lower resolution after an out-of-memory error.

Open a GitHub issue with Forge version/commit, operating system, GPU/VRAM/RAM,
checkpoint and LoRA filenames, image size, editing mode, reproduction steps and
the complete relevant traceback.

**Remove private paths, prompts, tokens and personal images before posting.**

## Isolation and Forge updates

Extension code stays in its own folder and hooks only `process_images_inner`
through a scoped, removable bridge. It does not rewrite Forge core files,
install shared dependencies or modify other extensions.

**Future compatibility cannot be guaranteed.** Keep a known-working backup
outside Forge before updating.

## License and credits

Extension code: [MIT License](LICENSE), based on [Adeliox's Klein Head Swap](https://github.com/Adeliox/klein-head-swap) and the owner's Enhanced v5.4 customizations. It uses [Alissonerdx's BFS workflow](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap), [Black Forest Labs' Klein models](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) and [Forge Neo reference inference](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References). Model, LoRA and detector weights keep their own upstream licenses; this repository's license does not relicense them. Fine-grid cleanup is adapted from Apache-2.0 ComfyUI-DeGrid; see [license](docs/DEGRID_LICENSE.txt) and [source](docs/FINE_GRID.md).

## Special thanks

Special thanks to:

- [r/sdforall](https://www.reddit.com/r/sdforall/) — community, support and inspiration
- [r/SECourses](https://www.reddit.com/r/SECourses/) — tutorials, guidance and community support
- [r/malcolmrey](https://www.reddit.com/r/malcolmrey/) — feedback and encouragement
- [Forge Neo (sd-webui-forge-classic, neo branch) by Haoming02](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) — the foundation this extension runs on, now with official Klein and Qwen 2.1 support
- [Adeliox](https://github.com/Adeliox/klein-head-swap) — original Klein Head Swap
- [Alissonerdx](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) — BFS (Best Face Swap) workflow and LoRAs
- Project Invisible extensions — memory policy, GPU compatibility and extension philosophy

Thank you to the wider Forge, Diffusers, Qwen, DeGrid and open-source communities.
