# Klein Head Swap Enhanced — 6.0.0-rc5

Head swapping and appearance cleanup for **Forge Neo with FLUX.2 Klein**, using the original picture as the body/scene reference and a selected headshot as the identity reference. This edition rebuilds the Enhanced v5.4 implementation around explicit settings, scoped Forge integration, reproducible reference selection, and measurable output checks.

**Release status: candidate; 104 offline tests passed with zero failures/skips.** Real local recognition-model smoke tests also passed five cases. The new checker has been exercised on the owner's existing rc2 outputs; those are not newly generated rc3 acceptance images. Live Forge browser/cancellation behavior, calibrated identity thresholds and improved generation quality still require validation. No Forge restart, diffusion generation or dependency upgrade was performed for this update. See [validation status](VALIDATION.md), [face-match auditing](docs/IDENTITY.md) and [the test procedure](docs/TESTING.md).

## Optional fine-grid cleanup (rc5)

Under **More options → Detail → Fine-grid cleanup**, enable **Automatically reduce fine 2-pixel grids** when needed. Start at the default 0.5 strength and compare at 100% zoom. It is off by default. Adapted from ComfyUI-DeGrid, this CPU filter detects repeating two-pixel grids, skips clean images and limits correction; it can still soften real repeating texture and does not remove tattoos or piercings. Protected edits keep this filter inside the edit mask. No model downloads or dependency changes. See [filter details](docs/FINE_GRID.md).

## rc4 batch-saving repair

The supplied September 12 log ends with `'H' format requires 0 <= number <= 65535` while Forge writes a JPEG grid. JPEG EXIF uses a two-byte segment length; long generation details can exceed its limit. This is a metadata-saving failure, not evidence of a GPU out-of-memory failure.

While a Klein session is active, a save callback calculates the same Unicode EXIF payload as Forge. If it exceeds the JPEG segment limit, that individual image/grid is saved as a lossless PNG with the full generation details and a unique `-metadata-` filename suffix. Normal JPEG saves, metadata-disabled saves and other extensions' independent generations retain their chosen format. The report and console explain the fallback. Global output settings are not changed. Optional Forge secondary JPEG exports may still report their own handled warning; the primary PNG remains saved.

Folder batches reuse the processing request. AutoNeg can write expanded negative text back into it; Klein now restores the original prompt and negative prompt at the end of each target, including list inputs and failure/cancellation cleanup. Existing saved presets and manually entered repeated words are not rewritten. Already-running sessions need an idle Forge restart to load this code.

The earlier head-boundary fix is retained: automatic sharpening does not stop at a head-shaped ellipse, and manual sharpening, tone mapping and grain run after compositing. These optional full-frame effects can change pixels outside a protected mask; leave them off when exact outside-mask preservation matters. This reduces a known artificial transition but is not proof that every generated hairline or neck seam is resolved.

Validation: 104 CPU tests passed, plus an isolated call to the installed Forge saving function preserved full oversized metadata and pixels. Fresh generation and live UI acceptance remain pending. The same log also contains separate Ideogram text-to-image-only model errors from an img2img request; this Klein repair does not add an img2img path to that model.

## What this version is designed to improve

- **Whole-body cleanup:** removal instructions cover tattoos, henna, body ink and remnants; piercings on all visible body regions; earrings and other jewelry; forehead sindoor, bindi, tikka, tilak, kumkum and related marks; and cross symbols. The original negative vocabulary is retained in a dedicated catalog.
- **Face-match auditing:** local CPU comparison against the selected headshot and full usable reference set, with a bounded background worker, explicit review statuses and JSON export.
- **Original proportions:** preserve the canvas aspect ratio, guide original head/body scale, associate the generated face with the selected target, and optionally correct measured face height and position with an isotropic transform.
- **Sharper faces:** compare original and generated face-region detail, apply bounded uniform sharpening only when needed, and report the result. Protected editing dedicates the sampling canvas to the head region and composites it into the original image.
- **Predictable prompts:** preserve user text, Unicode and unrelated LoRA tags; resolve appearance conflicts in one place; use the same planner for preview and generation; keep randomized choices tied to the image seed.
- **More reliable execution:** match 4B/9B adapters, handle old/new Forge reference options, cap both reference images, cache unchanged VAE encodes within a job, stop on preparation failure, and restore owned state in `finally`.

These are implementation features, **not a guarantee of perfect identity, complete removal, or recovered photographic detail**. A local contrast score cannot certify perceptual sharpness. The face detector does not detect tattoos, piercings or jewelry. Inspect final pictures at 100% before accepting them.

## Contents

1. [Compatibility and requirements](#compatibility-and-requirements)
2. [Install, update and rollback](#install-update-and-rollback)
3. [Quick start](#quick-start)
4. [Choose an editing mode](#choose-an-editing-mode)
5. [Removal policies and negative prompts](#removal-policies-and-negative-prompts)
6. [Head proportions and sharpness](#head-proportions-and-sharpness)
7. [Reference selection and memory](#reference-selection-and-memory)
8. [Prompts, adapters and presets](#prompts-adapters-and-presets)
9. [Control reference](#control-reference)
10. [Reports and troubleshooting](#reports-and-troubleshooting)
11. [Files, development and credits](#files-development-and-credits)

## Compatibility and requirements

The implementation targets **Forge Neo's native Klein reference-image path**, not generic Stable Diffusion img2img. The source inspected for this update was Forge Neo commit `efc42fe03739d0d8cda7de6e7bed2f8c1969a0c7`. Inspected local packages: Python 3.13.12, Gradio 4.40.0, Pillow 12.3.0, NumPy 2.3.5, MediaPipe 1.0.1 and Torch 2.13.0+cu130. These are an inspected environment, not a completed compatibility certification.

You need:

- A working [Forge Neo installation](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo). The neo branch now officially supports both FLUX.2 Klein and Qwen Image 2.1 / Qwen-Image-Edit, which are the two model families this extension auto-switches between.
- A Klein 4B or 9B checkpoint and a **matching** BFS face/head-swap LoRA. A Qwen-image-edit BFS LoRA is not interchangeable with Klein. Automatic selection requires a recognizable model size and a matching named adapter; manual selection can be used for a known custom adapter.
- At least one reference headshot and one img2img target image.
- Pillow and NumPy from Forge's environment. Optional face analysis additionally uses MediaPipe and its face-landmarker task model, or the legacy MediaPipe FaceMesh API when available.

Distilled Klein's published starting point is four steps and guidance 1.0; base checkpoints have a different inference regime. Use the checkpoint's own instructions. This extension leaves steps and sampler selection to Forge. Its **Positive-only** mode explicitly sets CFG to 1.0; new setups default to **Positive + Negative**, using at least CFG 1.1 to activate the requested negatives. This is an extension cleanup choice, not the published distilled-model default; compare both modes for your checkpoint. Explicit guidance choices in saved presets are preserved. [Klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), [Diffusers FLUX.2 documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2).

### Optional face detector setup

The existing local installation's detector is preserved. No automatic model download or package installation runs at extension import or during generation.

For a fresh installation, follow Google's Face Landmarker Python guide and place its compatible `face_landmarker.task` file at:

```text
<extension>/scripts/models/face_landmarker.task
```

Install MediaPipe in **Forge's own Python environment**, only when Forge and training are idle. Do not replace Torch, CUDA or Gradio just to install this extension. Restart Forge after dependency or detector changes. [Google Face Landmarker setup](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).

If the detector is unavailable, full-image swapping can use the best available reference fallback. Pose ranking and geometry/detail verification become unavailable. Protected editing then requires a custom mask; automatic quality checks still need a detected face. Strict quality rejection will reject unverifiable output.

## Install, update and rollback

**This is now a package, not a single-script update.** Copying only `klein_face_reference.py` will fail because the script imports `khs` beside the `scripts` directory.

1. Back up the complete existing extension outside Forge's active `extensions` folder.
2. Wait for training and any Forge generation to finish before restarting Forge. Files can be staged on disk while an existing process continues using its already loaded code.
3. Copy the full release structure into the existing extension folder. Keep exactly one active copy of the extension. Required files are `scripts/klein_face_reference.py` and the complete `khs/` directory; documentation and tests should travel with the release.
4. **Preserve** `scripts/custom_data.json`, `scripts/headswap_presets.json` and any existing `scripts/models/face_landmarker.task`. They contain local settings/model data. Example JSON files are provided for fresh installs; do not overwrite personal settings with examples during an update.
5. Restart Forge once it is idle and confirm the accordion displays **6.0.0-rc5**. Use a small single-output case before large batches. Follow the validation procedure before treating this candidate as stable.

Expected layout:

```text
klein-head-swap-master/
  scripts/
    klein_face_reference.py
    custom_data.example.json
    headswap_presets.example.json
    custom_data.json                  # local, preserved; optional
    headswap_presets.json             # local, preserved; created on save
    models/face_landmarker.task       # optional local model
  khs/
    __init__.py
    core.py
    data.py
    negative_catalog.py
    runtime.py
    vision.py
  tests/
  docs/
  README.md
  INSTALL.txt
  CHANGELOG.md
  VALIDATION.md
  LICENSE
```

To roll back, close Forge when idle, move the candidate folder outside active `extensions`, and restore the complete backup under the original folder name. Preserve newer personal JSON separately if desired. Start Forge and confirm the old version. Do not leave old and new folders both enabled. No Forge core files need to be replaced.

## Quick start

1. Load the intended Klein checkpoint and its matching BFS LoRA in Forge's registry.
2. Open **plain img2img** and upload the original body/scene image.
3. Enable **Klein Head Swap** and upload one or more sharp identity headshots. Prefer similar pose and lighting, enough visible face pixels, and an unobstructed face.
4. Leave the adapter on **Auto (match model)** or select its canonical registry name. Start at strength **1.0**. Keep the main denoising strength near the BFS workflow's starting value of **1.0**, then tune from actual comparisons. [BFS model card](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap).
5. Select **Full image edit** for tattoos or piercings elsewhere on the body. Leave the removal settings and **Removal priority** enabled. Keep original canvas, head-size checking and measured sharpening enabled initially. Keep experimental pixel resizing off to reduce hair/neck seam risk.
6. Choose a fixed main seed. Click **Check setup**. Inspect the selected face/reference and resolved prompt. A random main seed (`-1`) produces a clearly labeled seed-0 preview example, not a prediction of the upcoming random seed.
7. Generate one image after the system is idle. Open **Show last generation report**, inspect all visible skin at 100%, and check hair edges, forehead, ears, neck, shoulders, torso, hands and feet.
8. If a small head lacks detail but the rest of the image must remain unchanged, use **Protected head edit** and inspect its mask. For combined whole-body cleanup and protected head refinement, save the accepted full-image cleanup first and manually use that result as the target of a second protected pass. No automatic two-pass chain is hidden in this release.

Prefer PNG for final output when preserving exact pixels matters. JPEG compression changes pixels even when the in-memory protected composite preserves them.

## A simpler interface

The everyday panel contains the editing mode, headshot uploads, a cleanup/guidance summary, two quality controls and **Check setup**. Generation still uses Forge's main **Generate** button. **Setup preview** opens after checking and can be collapsed again.

| Location | Controls and purpose |
|---|---|
| Whole picture · swap + cleanup | Full-image editing, including tattoos and piercings outside the head |
| Head only · protect the rest | Protected head editing; **Edit head mask** opens its detailed controls |
| More options → Look | Six removal policies, guidance, instruction strength, hair/expression/styles and prompt controls |
| More options → References | Reference selection, target face, framing, adapters and memory limits |
| More options → Detail | Original canvas, head proportions, strict checks, head mask, finishing and experimental controls |
| More options → Presets and info | Save/load/delete setups, prompt preview and last generation report |

**More options** starts closed; specialized groups inside it are also collapsible. All 88 previous script arguments retain their positions; three new settings are appended (91 total). Hidden legacy compatibility arguments remain inert. No existing feature was removed to simplify the panel.

The main summary states whether negative guidance is active. A saved Positive-only preset remains Positive-only: choose **Look → Use negatives** to switch it explicitly. At CFG 1.0 the negative text remains available in metadata but is labeled inactive in the preview; the runtime also prints the effective CFG. Head-only mode shows that cleanup outside its mask needs a whole-picture pass.

## Choose an editing mode

| Mode | What is generated | What it can preserve | Appropriate use |
|---|---|---|---|
| **Full image edit** (default) | The full target/body/scene | Prompt-guided scene and anatomy; original canvas geometry when enabled | Removing marks or jewelry anywhere on the visible body |
| **Protected head edit** | A head/context crop, composited with a full-resolution mask | Original RGB values wherever the mask is exactly black, before optional full-frame finishing effects | Concentrating detail on a small head while keeping the scene/body outside the mask |

Protected mode is **not whole-body cleanup** with the automatic head mask. Tattoos and piercings outside the mask remain unchanged. An explicitly supplied custom mask can cover another/larger edit area, but the crop will expand to cover it, reducing the pixel budget dedicated to the head. It remains one generated crop, not independent region-by-region cleanup.

### Protected mask workflow

Select a target face number, click **Create editable mask from detected head**, and inspect the white region. It includes estimated hair, ears and a short neck transition, but it is a feathered ellipse, not hair segmentation. Paint white where changes are allowed and black where the input must remain. Keep the mask at the exact target image dimensions; empty or mismatched masks are rejected.

Expand the mask/context for large hairstyles or a head near the edge. Protect hands, glasses or foreground objects when appropriate. The automatic mask cannot understand these occlusions. Edge softness feathers the boundary; pixels in the gray transition are blended and can change.

Use plain img2img for protected editing. The extension rejects mixing its crop/composite workflow with Forge's inpainting mask. Full-image Forge inpainting requires turning **Keep original canvas** off, and remains an unvalidated combination in this candidate.

## Removal policies and negative prompts

Each category uses **Remove / Preserve / Use prompt or preset**. All six categories default to **Remove**.

| Category | Coverage |
|---|---|
| Tattoos | Permanent/temporary tattoos, henna/mehndi, body ink, remnants; face, neck, torso, back, arms, hands, legs and feet |
| Piercings | Ears, nose/nostril/septum, lip/tongue, eyebrow, chest/nipple, navel, dermal and surface jewelry, studs/rings/barbells |
| Bindi / sindoor | Bindi, sindoor in the hair parting, tikka/tika, tilak/tilaka, kumkum, forehead dots/ornaments and related terms |
| Earrings | Ear studs, hoops, dangling/drop/pearl/gold/silver earrings and ear ornaments |
| Jewelry | Necklaces, pendants, chains, bangles, bracelets, rings, anklets, nose pins, brooches and related ornaments |
| Cross | Cross symbols, crucifixes and cross ornaments on the body, clothing or elsewhere in the picture |

**Removal priority ON** makes removal the leading instruction. Conflicting dropdown modifiers are omitted with a report note. User text is preserved, with conflicts reported; the removal instruction explicitly takes priority. This preserves your prompt for review while avoiding silent destructive rewriting. Because the model still sees the original text, eliminate strongly conflicting wording for the cleanest experiment.

With priority OFF, a recognized affirmative request such as `wearing pearl earrings` can lift its removal policy. `studio portrait`, `walking across a room`, `no earrings` and `without tattoos` do not count as affirmative requests. The parser uses word boundaries and simple negation context; it is not a complete natural-language interpreter. **Remove jewelry or Remove piercings also requires bare ears**, even if Earrings was set to Preserve. Preservation always excludes items explicitly required to be removed by another category.

**Positive-only (fast, CFG 1.0)** uses explicit positive cleanup instructions and sets CFG to 1.0, even if an old preset enabled a CFG-boost checkbox. Your negative text is retained in metadata but typically has no effective CFG subtraction at 1.0. It is not silently advertised as active negative guidance.

**Positive + Negative** (the fresh-setup default) adds the full relevant negative catalog and uses `max(main CFG, 1.1)`. This enables an experimental stronger cleanup route and retains base-model CFG settings. It can cost more inference work and can change image quality; compare it with CFG 1.0 on the actual checkpoint. No universal quality improvement is claimed without visual tests.

The original additional negative preset remains available from `custom_data.json`. A selected negative preset is combined with the main negative prompt, and the optional additional-negative text is appended when enabled. Existing negatives are retained. Terms include waxy/plastic/over-smoothed skin, anatomical defects, duplicated heads and head-scale defects. See [the retained catalog](khs/negative_catalog.py) for exact vocabulary.

## Head proportions and sharpness

Three separate controls address different problems:

1. **Head-scale guidance** asks the model to keep the original head relative to the body. Strict and neck-match variants are wording choices, not geometric constraints enforced by the sampler.
2. **Keep original canvas** fits the target into an aligned sampling canvas without stretching, tracks its content box, and maps the generated picture back to original dimensions. The main Forge width/height's larger side sets the sampling-size budget. Increasing the output dimensions alone does not recreate missing detail.
3. **Check original head size and position** compares detected face height and center without moving finished pixels. Signed width/height/area differences are also reported. The separate experimental pixel-resizing control is **off by default** because transforming a finished head region can drag hair, neck and background pixels and leave seams. Explicitly enabling it permits only 0.85–1.15x resizing and a center move of at most 8% of face height or 3px, whichever is larger, followed by redetection and improvement/boundary checks. See [head-boundary safety](docs/IDENTITY.md).

The geometry check passes at **at most 8% detected face-height error**, with center displacement **at most 8% of target face height or 3 pixels, whichever is larger**. This is an operational tolerance, not “perfect anatomy.” It does not measure hair volume, skull shape under the hair, shoulder width, or neck thickness. Protected compositing preserves the body outside its mask; full-image generation can still change anatomy elsewhere.

**Match original face detail** compares grayscale face crops sampled consistently at 192×192 using a bounded Laplacian-based detail score. It measures the face and tries uniform unsharp-mask strengths 40, 70, 100 and 130 only when the generated score is below 95% of the original score. It selects the lowest tested amount reaching the target, or the best bounded improvement. It does not indiscriminately sharpen an already matching face.

The final image is measured again after compositing and effects. Texture noise can raise this score; sharpening cannot reconstruct real pores lost during low-resolution sampling. Judge the picture at 100% as well as reading the numbers. Grain, strong tone mapping and manual sharpening may work against natural-looking detail, so they default off.

**Reject outputs that fail head-size or detail checks** is optional and defaults OFF during candidate validation. When ON, an enabled check that fails or cannot be measured stops final output saving and gives a report. It does not reject tattoos or certify their removal. Earlier successful images in a batch remain saved if a later image fails.

Forge face restoration is disabled during enabled geometry/detail checks to avoid changing identity before comparison. Protected mode uses its own boundary matching and disables Forge color correction; strict quality mode also disables it to avoid intermediate color-correction output bypassing the final gate. Owned settings are restored after processing.

## Reference selection and memory

Up to 20 headshots are decoded. The target face selector is **1-based, largest detected face first**. The largest face in each reference is used for reference analysis. Use a single-person headshot for predictable identity. A nonexistent target face/explicit reference slot is rejected.

Reference scoring uses actual detected face crops, with approximate weights:

| Signal | Weight |
|---|---:|
| Pose compatibility (yaw/pitch/roll) | 45% |
| Local face detail | 20% |
| Face pixel size/framing | 15% |
| Face brightness and contrast compatibility | 20% |

Pose is derived from MediaPipe's transformation matrix where available, with a lower-confidence landmark estimate for the legacy API. Scores are heuristics, not identity similarity or calibrated probabilities. Full-frame detection is followed by overlapping upper-frame windows when no face is found; very small, obscured or unusual faces can still be missed.

- **Best match:** reuse the highest-ranked available reference.
- **Rotate good matches:** rotate through usable references within 15% of the best score when the top-cluster option is ON. It never forces three poor candidates into the pool.
- **Always rotation / scoring OFF:** follow upload order.
- **Manual slot:** select the uploaded slot explicitly; memory limits still apply.
- If no candidate passes usability checks, use the single best fallback and report the missing/weak detection.

**Head crop** keeps contextual hair/neck around the reference face. **Unmodified** retains the uploaded reference framing. **Match target framing** is experimental centered neutral padding; the corrected centering does not make this a guarantee of matching head scale.

Both target and identity references are bounded by a maximum side and a combined megapixel budget. They fit into multiples of 64 with padding instead of stretching. Total GPU memory and reported free memory can reduce the final cap. A small target head can request a larger side, but never bypass the final memory guard. These controls budget reference encoding; diffusion sampling can still run out of memory.

Successful native Forge VAE reference latents are cached on CPU within one generation session, keyed by image content and VAE identity, with a 256 MiB bound. A repeated target/headshot can reuse its encoding. The cache is cleared when the session closes. This is an implementation opportunity for reduced repeated work, not a measured speed-up claim.

Reference encoding retries once at a lower resolution after an out-of-memory error. An incomplete pair is never committed as a successful swap. Sampling errors are propagated; the extension does not automatically rerun an entire image or silently remove references.

## Prompts, adapters and presets

The extension preserves your full main prompt and unrelated `<lora:...>` tags. It replaces only its selected face-swap/character adapter tags, including recognized registry aliases. Tags are applied once at the effective reported weight. Auto adapter selection does not blindly prefer 9B when a 4B model is active.

An optional character LoRA has its own strength and trigger. It must be compatible with the loaded model. Use Forge's canonical registry names; spaces in folder names do not need to be “fixed” by renaming your model folders. Refresh the list after adding adapters.

The instruction strength slider changes prompt emphasis; it is not a pixel blend percentage. At 0, swap wording is disabled, but independent removal/scale/detail instructions and the selected adapter can still apply. Optional adapter boosting and tiny-head boosting default OFF and are capped at effective strength 2.0.

Style dropdowns accept multiple selections and custom text. With **Choose one per image** OFF, selected phrases are retained together; with it ON, one is chosen deterministically from the image seed. Random preset entries also use that local random generator. No global Python RNG reseeding is performed. Contradictory modifiers can still produce poor images, so review the resolved preview.

Klein's inspected text path uses Qwen3; the old T5/hard-512-token assumption was removed. The active tokenizer is used when available. Long prompts are retained and reported, rather than cut at an arbitrary character budget. Long context still costs memory and does not ensure better instruction following. [Klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).

### Saved settings and custom vocabulary

`scripts/headswap_presets.json` is saved atomically with a schema version. Loading a legacy flat preset file maps old booleans and obsolete resolutions into current settings. It does not rewrite your file just by reading it. A subsequent save preserves existing preset names and writes schema 2. Character adapter name, strength and trigger are included. Uploaded images/masks, transient reports and the enable state are excluded.

`scripts/custom_data.json` supports `hairstyles`, `hair_color`, `makeup`, `expressions`, `lighting`, `age`, `ethnicity`, `camera`, `earrings`, and a `negative_presets` object mapping names to text. Custom earring entries are now exposed. See [the example](scripts/custom_data.example.json). Unknown keys are ignored by the UI; invalid top-level or negative-preset structures are reported.

Existing presets may restore Preserve settings or old adapter names. Review all six removal choices after loading one if your intent is complete cleanup. The original user files are preserved during installation, not silently rewritten to new preferences.

## Control reference

| Control | Default | Meaning |
|---|---|---|
| Enable | Off | Participate only in img2img jobs when enabled |
| Face-swap adapter / strength | Auto / 1.0 | Resolve matching registered Klein BFS adapter |
| Target face | 1 | Largest detected target face first |
| Reference selection | Best match | Manual slot takes precedence; scoring OFF means upload rotation |
| Edit area | Full image | Whole-body cleanup remains possible |
| Original canvas | On | Restore original dimensions/aspect using tracked padding |
| Crop context / feather | 0.55 / 0.08 | Head context margin / mask-edge blur relative to detected size |
| Boundary color matching | 0 | Optional conservative color shift at protected blend edges |
| Head-scale wording | On, neck match | Request original scale and neck/shoulder junction |
| Head-size checking | On | Measure corresponding face without warping pixels |
| Experimental pixel resizing | Off | Limited post-generation transformation; may affect seams |
| Background face-match check | On | Advisory selected/best/median reference similarity and size report |
| Similarity review threshold | 0.363 | Benchmark starting point; not an accuracy percentage |
| Reference framing | Head crop | Preserve contextual hair and neck around the identity face |
| Instruction strength / order | 50 / extension first | Weighted instruction placement; not a visual blending ratio |
| Auto prompt / prevent extra head | On / On | Build wording when empty; ask for one replacement head |
| Six appearance categories | Remove | Full retained removal vocabulary |
| Removal priority | On | Leading removal instructions and conflicting-preset suppression |
| Guidance mode | Positive + Negative | Use at least CFG 1.1; fast mode forces 1.0 and labels negatives inactive |
| Additional negative text | Off | Append when enabled; main text is retained |
| Character adapter | None | Optional independent model-compatible adapter and trigger |
| Strict adapter checking | On | Reject unrecognized face-swap adapter family; known size mismatch is always rejected |
| Seed lock | Off | Reuse the first seed within the current target job |
| Maximum reference side | 1024 | One bound; memory and pixel budget can reduce it |
| Combined reference budget | 2.5 MP | Sum of the two encoded reference canvases |
| Small-head adaptation | On | Allow a 1280 request for heads below 220 pixels; final cap still wins |
| Encoding cache | On | Reuse unchanged references within the current job |
| Top-only rotation | On | Keep the good-match cluster when rotating |
| Adapter boost / tiny-head boost | Off / Off | Explicit bounded boosts, shown in the report |
| Latent sharpening / kernel | 0 / 3×3 | Experimental clamped latent contrast; leave off initially |
| No-reference diagnostics | Off | Generate without reference conditioning; not a head swap |
| Measured local sharpening | On | Match face detail score with bounded uniform unsharp filtering |
| Strict quality rejection | Off | Reject final output when an enabled numeric check fails or is unavailable |
| HDR-look / manual sharpness / grain / blur | Off / 0 / 0 / 0 | Optional finishing effects |

The “HDR-look” option is an **8-bit SDR tone-mapping effect**, not HDR10 mastering or a PQ/10-bit export. Background blur protects an estimated head ellipse in full-image mode; it does not segment the whole body. These labels replace inaccurate older HDR10 and background-segmentation claims.

## Face-match results

Enable **Check face match in the background** and open **Face-match results**. Refresh to see selected-reference, best and median similarity, size deviation and clear review notes. All processing stays local. The latest 300 results span folder-batch targets in the current instance; export them before restarting if needed. The advisory checker does not reject, overwrite or retry pictures. Its threshold and limitations are documented in [Face-match auditing](docs/IDENTITY.md). High agreement is a useful indicator, not proof of identity or image quality.

## Reports and troubleshooting

Stopping during reference preparation now follows a cancellation path. Completed outputs and their seeds/metadata are retained; stopping before the first output returns no images. Owned reference state is restored. A concise cancellation message replaces the duplicated error traceback seen in the supplied rc1 log. Actual model/preparation errors still fail visibly. This behavior passed offline lifecycle tests; live Forge cancellation and recovery remain pending.

**Check setup** reads the main plain-img2img image, prompt, negative prompt, CFG and seed. It shows face detection, the selected upload, reference scores, resolved choices, effective adapter weight and the planned prompt. It uses the same deterministic planner as generation. Batch/Inpaint inputs and prompt changes from other extensions can differ from this preview; the generation report is authoritative.

**Show last generation report** includes the selected reference/reason, detector status, prompt plan, encoded dimensions, fingerprints, cache hits, VAE encode count, memory retry, protected crop box and quality measurements/history. It reports errors as errors instead of pretending a swap completed. Effective settings, reference identifiers and quality values also enter Forge generation metadata. Save that metadata with the image for reproducibility; it does not embed original uploaded image files.

| Symptom | Next action |
|---|---|
| Missing `khs` import / old UI | Copy the whole package beside `scripts`, keep one active extension copy, restart when idle |
| Adapter missing/ambiguous/mismatched | Refresh the registry; select the canonical matching 4B/9B Klein BFS name |
| Active model is not Klein | Load Klein or disable this extension; force mode is deliberately not used to fake compatibility |
| No detected face | Use a larger/clearer headshot, check the optional task file and MediaPipe; use a custom mask for protected editing |
| Wrong person selected | Inspect target-face order; for multi-person scenes use protected masking for stronger spatial isolation |
| Tattoo remains on an arm/torso | Use full-image editing or an explicit mask covering it; inspect removal settings and conflicting text; compare dual guidance |
| Earrings remain despite Earrings=Remove | Verify the actual image, prompt and report; instructions can fail and there is no automatic jewelry detector |
| Neck seam / ghost hair | Keep experimental pixel resizing off; review hair/ear/neck mask coverage and foreground objects; use a closer pose match. Existing damaged outputs need a fresh generation/refinement, not just another resize |
| Face remains soft | Use a sharper reference, increase actual sampling pixels within memory limits, or use a protected crop; output resizing alone cannot restore texture |
| Geometry/detail gate rejects | Read the measured reason; choose a better reference or expand context; only disable strict rejection after reviewing the limitation |
| Out of memory | Reduce main sampling dimensions as well as reference limits; only reference encoding has a single reduced-size retry |
| Another extension changes the result | Validate Klein alone first, then enable other postprocessors individually; ADetailer/nested processing combinations are pending integration validation |
| Saved preset restores unwanted look | Review all appearance settings and current adapter after migration; save a new clearly named cleanup preset |
| Dark/harsh HDR look | Turn off SDR tone mapping or reduce its blend; there is no HDR10 output in this version |

Batch size is serialized to one image at a time while retaining the requested output count (`batch_size × n_iter`). Multiple distinct init images in one request are rejected instead of silently using the first. Use Forge's batch-file workflow to process separate targets sequentially. No-reference diagnostics cannot be combined with protected head editing.

## Files, development and credits

| File | Responsibility |
|---|---|
| `scripts/klein_face_reference.py` | Gradio UI, main-input capture and Forge script callbacks |
| `khs/core.py` | Normalization, migration, deterministic prompt planning, scoring, image geometry and caches |
| `khs/negative_catalog.py` | Retained original cleanup/quality vocabulary |
| `khs/data.py` | Existing built-in preset vocabulary |
| `khs/vision.py` | Lazy optional MediaPipe face analysis and bounded CPU results cache |
| `khs/runtime.py` | Scoped native Forge reference encoding, lifecycle bridge and output checks |
| `tests/` | CPU regression and stand-in integration tests; 104 tests passed |
| `docs/ARCHITECTURE.md` | Integration order, owned state and design limits |
| `docs/TESTING.md` | Regression, UI and real-generation acceptance procedure |

There are no automatic model downloads, external telemetry, remote image uploads, attention-kernel replacements or Forge core-file edits. Optional face-match model setup is explicit through `tools/setup_identity_models.py`, with pinned URLs and verified hashes. It uses Forge's existing model and LoRA infrastructure. API clients using positional script arguments retain the first 70 legacy positions; new fields are appended. Old dead controls remain inert compatibility slots; consult `core.ARG_KEYS` and `core.DEFAULTS` rather than guessing an argument order.

The source package excludes local model binaries, uploaded images, caches and personal JSON through `.gitignore`. The local GitHub source folder is a local copy; synchronizing files does not publish or push a GitHub repository.

Based on [Adeliox's Klein Head Swap](https://github.com/Adeliox/klein-head-swap), under the retained [MIT license](LICENSE), and the owner's Enhanced v5.4 customizations. It uses [Alissonerdx's BFS workflow](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap), [Black Forest Labs' Klein models](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), and [Forge Neo reference inference](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References). Model, LoRA and detector weights keep their own upstream licenses; the extension's MIT license does not relicense them.


Third-party filter code is adapted from Apache-2.0 ComfyUI-DeGrid; see [license](docs/DEGRID_LICENSE.txt) and [source/modifications](docs/FINE_GRID.md). The repository MIT license does not replace that component license.

## Special thanks

With gratitude to the communities and projects that made this extension possible:

- [r/sdforall](https://www.reddit.com/r/sdforall/) — Stable Diffusion community
- [r/SECourses](https://www.reddit.com/r/SECourses/) — SECourses community
- [r/malcolmrey](https://www.reddit.com/r/malcolmrey/) — community
- [Haoming02's sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) — Forge Neo host providing the Klein and Qwen-Image-Edit reference inference this extension is built on
- [Adeliox](https://github.com/Adeliox/klein-head-swap) — original Klein Head Swap
- [Alissonerdx](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) — BFS (Best Face Swap) workflow and LoRAs
- [Project Invisible](https://github.com/) — memory policy, GPU compatibility and extension philosophy that guided this build
- [ComfyUI-DeGrid](https://github.com/) — Apache-2.0 fine-grid cleanup filter
- Google MediaPipe — face landmarker and detector models
- And everyone else in the Stable Diffusion and Forge communities whose workflows, bug reports, tests and discussions contributed.
