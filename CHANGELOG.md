# Changelog

## 1.0 — 2026-09-25 · Universal Head Swap

**One extension. Two models. Zero toggles.**

- 🔁 **Dual-model engine** — FLUX.2 Klein 4B/9B *and* Qwen Image Edit 2.1. Switch the checkpoint; the extension auto-switches its reference plumbing, LoRA matching and prompt formatting.
- 🎭 **Full Klein feature set on Qwen 2.1** — protected head edit, removal policies, face-match auditing, presets, fine-grid cleanup, quality gates. Nothing dropped.
- 🧬 **Family-locked adapters** — Klein and Qwen BFS/character LoRAs can never cross-load. Auto-match works in both modes.
- ⚡ **Project Invisible memory policy** — device-aware probing, bounded reference budgets, OOM retry, 256 MiB encode cache. NVIDIA, AMD ROCm and CPU. Any VRAM, any quantization.
- 🖥️ **Cleaner UI** — dedicated **Adapters** and **Memory** tabs in More options.
- ✅ **119 offline tests passing**; no Forge core edits — delete the folder and Forge is stock.

## 6.0.0-rc5 — 2026-09-21

- Added optional, default-off automatic DeGrid cleanup with clean-image bypass, bounded correction and adjustable strength.
- Run cleanup before resize/sharpen and preserve protected compositing; final quality checks follow all effects.
- Appended settings for backward compatibility; no model/dependency changes.


## 6.0.0-rc4 - 2026-09-12

- Fixed JPEG grid EXIF overflow with a scoped, lossless PNG fallback preserving full metadata and using a unique filename.
- Restored original scalar/list prompt inputs between folder-batch targets to prevent expanded AutoNeg text from carrying forward.
- Preserved the newer rc3 uniform-sharpening and post-composite finishing fixes.
- Passed 104 CPU tests and an isolated installed-Forge saver round-trip. Fresh GPU generation and live Forge lifecycle acceptance remain pending.


## 6.0.0-rc3 — 2026-09-11

- Added local SFace/YuNet identity comparison, selected/best/median and per-reference scores, reference consistency notes, signed geometry results, bounded background auditing and a simple refresh/export panel.
- Added explicit checksum-verified model setup and upstream notices; no import-time downloads or dependency changes.
- Kept 88 existing positional arguments and appended three settings.
- Made finished head pixel resizing opt-in after inspecting its seam risk in existing user outputs; tightened optional scale/translation limits and widened its feather.
- Fixed the reported inpainting-style seam around the head: automatic detail matching no longer confines its unsharp mask to a head ellipse (which sharpened hair, neck and background inside the ellipse only, leaving a halo ring), and manual sharpening, tone mapping and grain now run on the finished frame instead of on the protected crop, so they no longer stop at the head mask boundary.
- Passed 97 offline tests, including a new check that the detail correction is applied uniformly. Live image comparison of the fix remains pending.


## 6.0.0-rc2 — 2026-09-11

- Simplified everyday panel; grouped advanced controls into Look, References, Detail, and Presets and info. Kept all 88 argument positions and existing features.
- Added editing-mode guidance, a mask-editor shortcut and visible negative-guidance status.
- New setups activate negative guidance with at least CFG 1.1; explicit saved choices remain unchanged.
- Separate cancellation from genuine errors, retain completed images and metadata, and avoid duplicate failure logging.
- Passed 71 distinct offline checks, including real Gradio construction and cancellation/result lifecycle tests. Live rc2 browser/generation and image-quality acceptance remain pending.

## 6.0.0-rc1 — 2026-09-11

Implementation candidate, based on the owner's Enhanced v5.4. Regression and visual testing deferred while AI Toolkit trains.

### Added

- Full-image cleanup as the default, six explicit appearance policies and removal-priority control.
- Retained original negative vocabulary, expanded positive whole-body/forehead removal wording and visible conflicts.
- Protected head crop, editable mask, context/feather controls, original-pixel composite and optional boundary color matching.
- Original-canvas aspect preservation, generated-face association, bounded head-scale/position correction and final geometry reports.
- Measured bounded local sharpening, final detail reports and optional strict rejection of unverifiable/failed numeric checks.
- Canonical adapter resolution with automatic matching to known Klein 4B/9B size, character-adapter persistence and reference fingerprints.
- Bounded CPU caches for analysis and per-job native VAE references; shared pair budget and one smaller reference-encoding retry after OOM.
- Modular core, vision and runtime package; prepared tests; detailed installation, architecture and validation documentation.

### Repaired in code

- Modern `klein_do_reference` and legacy inverse-option handling, with scoped restoration.
- Failures swallowed by Forge callbacks: propagate before init/sampling/final saving via an outer callback guard.
- User prompt truncation and removal of unrelated LoRA tags; preserve text and replace only owned adapter tags/aliases.
- Negation and substring errors in appearance matching; unify previously competing ban systems.
- Fast mode accidentally inheriting legacy CFG boost; explicit effective CFG policy.
- Off-center padding, incorrect face-region brightness coordinates and forced low-quality top-three candidates.
- Manual/automatic reference sizes bypassing limits; apply final side and combined-area checks to both references.
- Preview using unrelated/random inputs; share the planner with actual main img2img inputs and label random-seed examples.
- Partial preset serialization and non-atomic writes; keep character settings, migrate legacy values and preserve new selection policies on reload.
- Hidden unused custom earring choices, ignored multi-select phrases, inaccurate HDR10/segmentation labels and single-file installation instructions.
- Gradio 4.40 gallery compatibility, PNG previews/masks and full-size mask validation.

### Compatibility and limits

- Requires copying `khs/` with the main script; keep one active extension directory.
- Keeps original 70 script-argument positions; obsolete controls are inert.
- Preserves local custom/preset JSON and detector weights during installation.
- No claim of universal removal, exact anatomy, restored photographic detail, runtime speed or GPU stability before acceptance tests.
- No Forge core-file edits, automatic model downloads or GitHub publishing.

## September 26, 2026 runtime fixes

- Qwen references retain their original colors and stay in CPU memory until Forge needs them.
- Automatic matching checks the loaded architecture, including Klein 4B/9B, before stale preset flags. Keep the adapter dropdown on Auto; matching runs after Forge loads the selected checkpoint.
- Text-only Qwen is rejected: an Image Edit checkpoint and Forge edit mode are required.
- Reinstalling the generation bridge returns the existing wrapper correctly.
- Existing advanced controls and Forge-managed model offloading remain in place. Quantization support follows the installed Forge loader; every quantization has not been tested.
- CPU regression checks do not establish visual quality or GPU performance. Real generations on all three models are still required.

## Dedicated Qwen 2.1 integration repair

The Project Invisible Qwen 2.1 engine bypasses Forge's standard generation callbacks. Head Swap now has an explicit bridge into that engine, so its references, prompts, BFS adapter, character adapter, finishing controls, protected mask and seed lock are actually used.

- Automatic adapter selection follows preset/checkpoint changes. It prefers `bfs_head_v1_qwen_2.1` for Qwen and `bfs_head_v1_flux-klein_9b_step3500_rank128` for Klein 9B, including adapters discovered in subfolders. Turn off automatic selection to use a custom adapter.
- Up to 20 headshots form the selection pool; each output sends its target and selected headshot to Qwen. Best-match, manual-slot and rotation controls remain available. This avoids encoding all 13 headshots for every output.
- Qwen protected-head mode conditions on the complete scene and composites only the chosen head region. It handles RGBA output and preserves original output dimensions. Native inpaint masks must be cleared; use the extension's protected mask control.
- Qwen BFS, character and the installed Viggle v0.2.1 Turbo adapter were tested together with INT8 ConvRot on an RTX 5090. Turbo keeps CFG 1; its negative prompt is inactive. Klein turbo prompt tags also keep CFG 1. Quantization remains owned by each engine; other quantizations and a real Klein generation were not GPU-validated in this repair.
- Saving now resolves an output-folder fallback, honors batch folder/name overrides, checks the written file and reports its path. Explicitly disabled saving remains disabled and is reported.
- Optional CPU identity checks are now connected to completed images and shown in the existing generation report. They are advisory; a passing score does not guarantee likeness. The real protected-head test passed the selected/median identity thresholds and face-height/center check; width/reference-consistency warnings still require visual review.

Restart Forge completely after the current batch finishes to load both updated extensions. A browser refresh alone does not load Python changes.
