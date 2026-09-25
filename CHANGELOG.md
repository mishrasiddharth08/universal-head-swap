# Changelog

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
