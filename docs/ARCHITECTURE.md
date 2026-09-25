# Integration design

This file describes the candidate implementation, not results of completed integration tests.

## Lifecycle

The UI's `before_process` records normalized settings on the request. Forge loads its checkpoint in `process_images`, then calls `process_images_inner`. The extension wraps that inner call without changing its signature or replacing the original implementation. Disabled requests are forwarded unchanged. Enabled requests validate Klein, a single target and adapters before reference/global option mutation.

A `Session` snapshots owned state, switches the appropriate reference option, serializes output batches, prepares detection/crop geometry and forwards execution to Forge. The `process` callback resolves every prompt with its actual seed. `process_batch` selects and encodes the target/reference pair. Forge's native `encode_first_stage` determines the correct reference latent representation. The finished image passes through optional geometry/detail correction and compositing before Forge's normal save.

Cancellation during reference preparation uses `GenerationCancelled`, separate from actual failures. The outer script guard captures completed images after all final image callbacks, together with their seeds and infotext. The bridge returns a partial `Processed` result on cancellation, including an empty result when no image completed; actual exceptions still propagate. Adapter cleanup runs for failures and cancellation.

The bridge's `finally` closes the session on success, cancellation or exception. It restores the original reference option, model reference/initial latents, dynamic reference state and request attributes it changed. It drops its bounded CPU latent cache. It does not change `default_ref_method` or install attention implementations.

## Why a callback guard exists

The inspected Forge ScriptRunner catches and logs ordinary extension callback exceptions. Letting reference preparation merely raise inside such a callback can allow sampling to continue without the intended references. `ScriptGuard` forwards to the original runner, then raises an extension-recorded failure **outside** its swallowed callback. Additional guards check before init, conditioning and sampling. The final-composite callback is guarded before the host save call. Exceptions still flow to Forge's normal outer error handling.

Other extensions can install wrappers or mutate request state; their combinations require integration tests. `_ad_inner` and high-resolution nested requests are skipped as separate extension jobs, but nested processors share host/model state. ADetailer and other recursive processors are not certified here. Start validation with other postprocessors disabled.

## Reference transaction and cache

1. Decode every upload and name the failing slot if decoding fails.
2. Select one identity input with the explicit policy/manual override.
3. Prepare target/reference canvases under the same final side and combined area limits.
4. For each image, consult a CPU LRU keyed by VAE identity and SHA-256 image content.
5. Encode missing entries using native Forge into an isolated temporary reference list. Capture exactly one result. Always restore the temporary list/initial-latent/reference-flag state.
6. Only after both succeed, commit the complete pair as target then identity. Clear prompt conditioning cache so the next conditioning sees the pair.
7. On an encoding OOM only, clear the local cache and retry once at reduced dimensions. Do not swallow a second failure or retry sampling automatically.

Cached tensors are CPU tensors just as the inspected native Klein path stores its references. The 256 MiB limit bounds retained latents, not total generation memory. Dynamic host memory availability is an estimate, not a reservation against another program.

## Geometry and image ownership

Full-image mode can fit the source into a multiple-of-64 sampling canvas and remember the unpadded content box. The larger main width/height defines the requested sampling side. Output is mapped through the inverse content box and resized to source dimensions. This prevents aspect stretching; it cannot recover details absent at the sampling resolution.

Protected mode derives a full-resolution mask, crops a padded context region, generates that crop and maps it back onto the original pixels. Face geometry is translated into crop coordinates before checking/correction. Corresponding output faces are associated by proximity rather than by their potentially changed size ordering. Pixel correction is now opt-in and off by default because it can move surrounding pixels. When enabled, it is isotropic and locally feathered, with tighter range/translation/boundary checks and redetection. It does not solve full anatomical registration or hair segmentation.

After compositing, the final detected face is measured again. Geometry tolerances and local-detail checks are recorded even when strict rejection is off. Zero-valued mask pixels come from the original RGB buffer. Compressed file formats and downstream extensions can change those values after this extension's callback.

## Prompt ownership and migration

`core.build_plan` owns extension instructions, selected adapter tags, optional character trigger, deterministic dropdown choices, effective appearance policy and CFG decision. It preserves user text and unrelated LoRAs. Preview and generation call this same function; preview uses an explicitly labeled example seed when the user requests a random seed. Main host scripts may subsequently alter prompts, so the preview is not a promise of another extension's output.

The 70 original positional arguments remain in order; new arguments are appended. Deprecated force-mode/old-ban/CFG controls remain inert compatibility positions. JSON schema 2 stores only meaningful persistent settings. Legacy migration applies only legacy selection overrides to legacy files, so newly saved Best-match settings do not become rotation settings on reload. Writes use a same-directory temporary file, flush/fsync and atomic replacement.

## Intentional limits

- No learned tattoo, jewelry or piercing detector, no automatic removal verification and no iterative removal pass.
- No identity-recognition score or claim that a pose/detail score measures identity fidelity.
- No full-body segmentation, hair matting or guaranteed exact head/body anatomy.
- No super-resolution model or invented-detail reconstruction in the automatic sharpening step.
- No guaranteed speed-up, VRAM ceiling or real-image success rate before benchmarks.
- No persistent latent cache across jobs/models; no network downloads on import.
- No UI restart, diffusion model import or regression execution during the owner's training window.

Potential later improvements should be supported by the acceptance images: segmentation-guided masks, region-specific cleanup with explicit masks, calibrated face/body landmark alignment, and a separately budgeted refinement pass. They are not represented as implemented features.

## Background identity audits

`khs.identity` owns a lazily started per-instance CPU executor, two in-flight slots, 300 synchronized history records and 128 cached descriptors. The outer final image-callback guard submits immutable bounded samples with unique IDs and fingerprints, before saving metadata containing an audit ticket. Workers do not modify Forge requests. Reports are refreshed/exported explicitly in Gradio. Unload cancels queued futures; already running native inference finishes independently. See [the detailed contract](IDENTITY.md). First 88 arguments remain fixed, with identity_check, identity_threshold and geometry_correct appended.
