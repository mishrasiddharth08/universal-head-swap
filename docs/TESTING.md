# Validation procedure

**Do not run this procedure while the owner's AI Toolkit is training.** Offline checks have passed as recorded in VALIDATION.md. Browser, live Forge lifecycle and image acceptance remain pending. Do not repeat passing checks unless code changes or new evidence justify it.

## Before starting

Read `VALIDATION.md`. Use read-only process/status/log evidence to establish that the intended training run has finished. A temporarily quiet GPU, a checkpoint file, or a training UI merely being open/closed is not sufficient evidence. If status is uncertain, leave model loads, restarts and generation pending. Do not stop, suspend or reconfigure training to make testing possible.

After confirming idle, first run the bounded CPU suites. Only then consider an isolated Forge smoke test or an idle, controlled restart. Never kill an active Forge generation or rely on an unverified idle API: this installation previously returned 404 for `/sdapi/v1/progress`. Its running UI may still contain the old code until restarted.

## Stage 1 — CPU regression suite

From this extension's root, in PowerShell using Forge's existing Python:

```powershell
& 'G:\FORGE UI NEO\sd-webui-forge-classic\venv\Scripts\python.exe' -B -m unittest discover -s tests -v
```

For another installation, substitute its interpreter. These tests use synthetic images and host stand-ins. Core tests need Pillow and NumPy. The UI construction test uses installed Gradio when available and explicitly blocks Torch/Forge model imports. Missing Gradio is reported as a skip, not as UI verification. No detector model or weights are loaded by the tests.

Coverage includes long/Unicode prompts and external LoRAs, effective CFG, negation boundaries, removal priority and retained vocabulary, deterministic choice without global RNG mutation, crop coordinates and alignment, budgets, pixel-preserving compositing, geometry/quality failure handling, preset migration and atomic-save failure, adapter model-size matching, reference-option polarity, cache behavior, cancellation, swallowed callback failures and scoped cleanup.

Record the exact command, interpreter, counts, failures/skips and date. Fix failing behavior and rerun relevant tests; do not call an unexecuted check “passed.”

## Stage 2 — Forge UI and lifecycle

Confirm the loaded accordion displays 6.0.0-rc5. Review console startup errors. Verify gallery upload/removal, manual selection, target capture, preview, mask creation/editing and preset save/load/delete with temporary preset names. Preserve the user's existing JSON; keep backups before a save/migration test. A standalone Gradio constructor test does not validate browser interaction or Forge callbacks.

Exercise disabled mode first: prompts, CFG, batch count and reference settings must remain normal. Enabled wrong-model and missing-adapter requests must fail clearly before sampling. Check modern `klein_do_reference`; the legacy polarity is covered by stand-ins unless an actual compatible older host is available.

For enabled jobs verify prompt order, actual LoRA activation, exactly two references, and correct target-first order. Force a controlled reference failure in an isolated test to confirm sampling/save never proceeds as a successful swap. Verify cleanup after success, failure, interruption and cancellation, then generate a normal disabled image and compare settings. Stop once during first reference preparation and once after completed images: expect no duplicate traceback, zero images in the first case, and only completed images with correct seeds/metadata in the second. Check the four More options tabs, both editing modes, the Edit head mask shortcut and guidance summary after loading a saved preset.

## Stage 3 — Small real-image acceptance set

Use authorized local photos of adults or synthetic adults. Do not publish the inputs/results. Start with one image per case and a fixed seed. Use a compatible local checkpoint/adapter; avoid downloading model weights as a shortcut. Keep other face-restoration/postprocessing extensions off for the first comparison.

| Case | Inspect |
|---|---|
| Close frontal portrait | Identity, eye/nose structure, skin texture and sharpness at 100% |
| Three-quarter/profile head | Pose ranking, ears, hairline and nose silhouette |
| Distant/small head | Full mode versus protected crop at the same declared sampling budget |
| Large hair / foreground hand | Edited mask extent, seams, foreground-object preservation |
| Tattoos on neck/arm/torso/leg | Full-image removal at every visible marked region; no blurred ink remnants |
| Nose, ear, chest/navel/dermal jewelry | All visible piercings removed; natural skin reconstruction |
| Sindoor/bindi/tikka | Forehead and hair parting clean; original texture and hairstyle preserved |
| Multiple people | Selected target identity and corresponding-face association; other people unchanged as intended |
| Portrait/landscape odd dimensions | Original output dimensions/aspect and head-center placement |
| Batch of four same inputs | Correct count, seed policy, rotation policy and reused encode counters |

Make side-by-side comparisons of input, baseline v5.4 (only if safely available in isolation) and candidate with the same checkpoint, adapter, seed, sampling size, steps, sampler/scheduler and CFG. Compare positive-only CFG 1.0 against dual guidance separately; do not label different-settings output a controlled improvement.

Measure/report detected face-height ratio, center error, original/output detail scores, exact outside-mask differences for PNG protected output, elapsed time and peak VRAM when safely measurable. Inspect neck/shoulder anatomy and hair volume visually because the automatic face-height metric does not cover them. Never claim complete tattoo removal from the numeric quality report.

## Stage 4 — Release decision

Update `VALIDATION.md` with actual outcomes and attach private sample paths or contact sheets to the local work report. Keep image metadata and note unresolved cases. Synchronize corrected managed code/docs/tests to both the live extension and local GitHub source copy, preserving personal JSON and models. Promote a candidate only after CPU, loaded UI, error-cleanup and real-image criteria pass; otherwise retain the release-candidate suffix and the relevant pending/failing items explicit.

If live training status or a safe validation environment cannot be established, finish the checks that are actually safe and record the remaining dependency. Do not create a false success entry to close the task.

## rc3 face audit and seam checks

The 104-test offline suite and five real-model smoke cases are recorded in VALIDATION.md. For live validation, verify one selected person in a multi-person target, pending/completed result refresh, history across folder-batch targets, export contents, missing/corrupt model reporting, and no identity descriptor/image leakage into exports. Compare a fixed-seed generation with pixel resizing off and on; inspect hair, neck, ears and nearby straight background lines at 100%. The default off path must not call the pixel-warp function. Reports are advisory, so low identity scores must not delete completed images. Preserve user files and sample outputs.
