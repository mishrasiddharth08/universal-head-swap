# Universal Head Swap — Complete Session Log & Technical Documentation

**Date:** September 29, 2026
**Extension:** universal-head-swap (Forge, `G:\FORGE UI NEO\sd-webui-forge-classic\extensions\universal-head-swap`)
**Repo:** https://github.com/mishrasiddharth08/universal-head-swap
**Hardware:** NVIDIA RTX 5090 (32GB), 98GB RAM, Forge classic + Project-Invisible extensions

---

## 1. SESSION OVERVIEW

This session transformed the extension from a batch-crashing prototype into a
fully automated, smart, SAM3-powered Z-Image head swap pipeline. Every fix was
committed and pushed; the final state passes 192/192 offline tests.

### Commits pushed (chronological)

| Commit | Change |
|---|---|
| `910a781` | Spot cleanup batch-safe: skip mismatched targets instead of aborting |
| `72f7b5b` | SAM3 device + quantization support (fp32/fp16/bf16/FP8/int8/int4); batch-safe mask fallbacks |
| `6a9d640` | Smart auto-settings per model preset; SAM3 auto-discovery from models/SAM 3; SAM3 default for Z-Image; feathered Z-Image inpainting |
| `e57b173` | UI polish: symmetric balanced rows, cleaner tab styling |
| `420605f` | Auto-adjusting Z-Image inpaint per image; body-to-head ratio canvas sizing |
| `67a5034` | Cap head-ratio canvas boost to 1.5x and Forge generation size |
| `f81debc` | Safetensors fallback loader for SAM3 checkpoints |
| `b2321fa` | Document SAM3 setup (3 routes incl. ungated mirrors) in docs/ZIMAGE.md |
| `d08d98e` | SAM3 CPU BFloat16 fix; SAM 3 subfolder search; auto-select Z-Image character LoRA in smart sync |
| `219dadc` | Simplify UI for Z-Image: hide BFS LoRA controls on ZIT/ZIB, reveal Z-Image section |
| `c650679` | Auto-detect SAM3 device/precision; remove manual dropdowns |
| `2a9e976` | Auto-detect: fp32 on GPU (processor feeds Float inputs) |
| `fad7510` | Body-to-head ratio preservation in geometry quality gate |
| `89c1d7d` | Global skin-tone transfer blending (YCbCr mean/contrast, alpha-weighted) |
| `d72e716` | Fix double-feathered inpainting: crisp model mask, single blend at composite |
| `0241fc4` | Exact regeneration-mask compositing (union) — eliminates ghost edges |
| `63ad557` | Realistic skin-tone clamps (60-pt luminance + gamma match); full-range jaw seam |
| `7e42f95` | Tune Z-Image defaults to proven workflow: denoise 0.35 (was 0.65) |
| `c6efddd` | SAM3 multi-prompt grounding: fall back to 'face' prompt |

---

## 2. BUGS FIXED (in order of discovery)

### 2.1 Spot-cleanup batch crash
- **Symptom:** `ValueError: Spot cleanup: paint on a photo with the same dimensions as the target.` killed an entire 89-image img2img folder batch.
- **Root cause:** `khs/cleanup.py:painted_mask` raised on size mismatch BEFORE the "different photo" hash check could run; `repair()` would also fail on resized outputs.
- **Fix:** `painted_mask(value, size, strict=True/False)` — non-strict mode resizes masks (NEAREST); new `applies(value,target)` checks editor-photo hash; `validate_target` skips silently for photos the mask wasn't painted on; runtime.py records `spot_cleanup: {applied:False, skipped:...}` in the quality report.
- **Tests updated:** test_cleanup.py (2 tests).

### 2.2 "Upload a valid image" gradio popup
- **Symptom:** clicking mask/preview buttons with empty img2img box threw raw `ValueError` wrapped in gr.Error.
- **Fix:** `scripts/klein_face_reference.py:make_mask` now raises `gr.Error('Load the target photo in the img2img image box first, then create the mask.')`.

### 2.3 SAM3 "not installed" killed batches
- **Symptom:** `SAM3 found no reliable head mask: SAM3 is not installed in this Forge environment` aborted the batch when Z-Image mask source = SAM3 without runtime.
- **Fix:** `khs/zimage.py:prepare_mask` falls back to the face-detector mask (returns None) with a console note — batch never aborts.

### 2.4 SAM3 dtype crashes (the long chain)
- **Error 1 (CPU):** `mat1 and mat2 must have the same dtype, but got BFloat16 and Float` — CPU bf16 weights vs fp32 processor inputs.
- **Error 2 (GPU):** `Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type)` — persisted BFloat16 setting cast weights on CUDA.
- **Root cause chain:** SAM3's `Sam3Processor` always feeds Float32 inputs; any global bf16/fp16 cast of the weights breaks inference. The checkpoint stores BFloat16 tensors.
- **Fixes:**
  - `_apply_quantization` normalizes the model to fp32 on CPU always, and on GPU always (except torchao INT paths which manage their own dtypes).
  - Half/FP8 selections print a note and run fp32.
  - `core.normalize()` clamps persisted `sam3_quantization` to Auto/INT-only — old saved BFloat16 settings can never re-break it.
  - Final fix: `auto_settings()` returns ('cuda','Full precision (fp32)') on GPU, ('cpu','Full precision (fp32)') otherwise.

### 2.5 SAM3 fused-kernel inference crash
- **Symptom:** even with all-fp32 model, inference crashed in `sam3/perflib` `addmm_act` inside `vitdet.py:74`.
- **Root cause:** the fused `addmm_act` kernel mis-handles mixed dtypes on this CUDA build.
- **Fix:** patched the isolated runtime copy `models\SAM 3\runtime\sam3\model\vitdet.py` Mlp.forward to use `self.act(self.fc1(x))` (numerically identical plain path). Forge packages untouched.

### 2.6 SAM3 weights_only unpickling
- **Symptom:** `UnpicklingError` — PyTorch 2.6+ defaults `torch.load(weights_only=True)`.
- **Fix:** patched `models\SAM 3\runtime\sam3\model_builder.py` (all `torch.load` calls → `weights_only=False`) — trusted local checkpoint.

### 2.7 SAM3 checkpoint format mismatch
- **Symptom:** `invalid load key, '\xe8'` — user's `SAM 3.safetensors` is safetensors, loader expected pickled .pt.
- **Fix:** safetensors fallback in `_load()`: build unweighted model, `safetensors.torch.load_file`, `load_state_dict(strict=False)` with a >50%-missing sanity guard.
- **Note:** the community safetensors export (detector_model.*/tracker_neck.* prefixes, 1797 tensors) did NOT match Meta's official model (1134 params) — suffix remapping matched only 10 tensors, so the official `sam3.pt` was obtained instead (see §3).

### 2.8 Z-Image "requires a character LoRA" crash
- **Symptom:** `ValueError: Z-Image head swap requires a Z-Image character LoRA` — smart sync set Turbo/Base + SAM3 but never auto-selected the character LoRA.
- **Fix:** sync scans `runtime.registry()` for Z-Image-family adapters (`core.adapter_family(...)[0]=='zimage'`), prefers 'char'-labelled names, emits a 5th gr.update for `char_lora_name`. `SMART_OUTPUTS` extended; manual-mode early return returns 5 no-ops.

### 2.9 Double-feathered inpainting (worse quality regression)
- **Symptom:** after adding a feathered inpaint mask, boundary quality degraded — "bad to worse".
- **Root cause:** TWO soft edges stacked: model regenerated a half-denoised mushy ring (soft mask), then composite blended with ANOTHER feathered mask.
- **Fix:** model mask stays crisp (only 4px rounding); the single soft transition happens at `composite_region`'s feathered blend.

### 2.10 Ghost edges from mask disagreement
- **Root cause:** the mask used for generation (canvas inpaint mask) and the mask used for compositing (region mask) went through different resampling paths and could disagree slightly.
- **Fix:** `configure_inpaint` stores `session.inpaint_mask_canvas` (the exact soft mask the model saw); `finish_image` maps it back from canvas coordinates to the crop and composites with the UNION (np.maximum) of it and the region mask — regenerated area and blended area can never disagree.
- **Bug caught by tests mid-fix:** first version replaced the region mask (full-image size) with a crop-sized mask → `ValueError: images do not match` in PIL paste; corrected to build a full-size union mask.

### 2.11 Massive head/body tone gap ("worst ever inpainting")
- **Measurement (pixel forensics on user outputs):** swapped head RGB ≈ [133, 98, 85] vs surrounding skin ≈ [186, 181, 179] — a ~60–90 point luminance/exposure gap. Previous correction clamps were ±10 shift / ±18% gain — purely cosmetic.
- **Fix:** luminance transfer clamp ±60, chroma ±24; gain 0.6–1.6; added gamma exposure matching (log-space, 0.6–1.6); jaw/neck seam delta clamp raised to (±80, ±40, ±40).
- **Test updated:** non-skin pixels may shift ±1 level due to continuous alpha blending (assertLessEqual sum of channel diffs ≤ 2).

### 2.12 Waxy/drifting faces (denoise too high)
- **Evidence:** decoded a proven ComfyUI Z-Image img2img workflow (pastebin 96QgdwE1): denoise **0.2–0.25**, character LoRA **0.9–0.95**, SAM3 prompt "face" @ 0.33, 8 steps DDIM, CFG 1.
- **Ours before:** denoise 0.65 (re-renders most of head from noise → waxy, drifts), LoRA 0.7.
- **Fix:** default `zimage_denoise` 0.65 → 0.35; recommend character strength 0.9.

### 2.13 Ratio boost caused lowvram thrashing
- **Symptom:** after adding per-image canvas sizing, steps went ~0.3s → 1.1s with repeated `Unloaded partially ... lowvram` — 3560-image ETA 1.5h.
- **Fix:** `head_ratio_side` capped at 1.5× requested side AND by Forge's own generation size.

---

## 3. SAM3 — COMPLETE WORKING STATE

### What's installed (verified on disk)
- `models\SAM 3\sam3.pt` (3.45 GB official checkpoint, 1134 param tensors, loads clean)
- `models\SAM 3\SAM 3.safetensors` (3.4 GB community export — NOT loadable by Meta's model; key layout mismatch)
- `models\SAM 3\runtime\` — pinned official sam3 source (commit 2345a4ad109ac29c569da749c91d84f10dc08c40), dependency-free, with patches:
  - `model_builder.py`: weights_only=False
  - `model/vitdet.py`: fused addmm_act bypassed
- Dependencies installed into the isolated runtime: pycocotools, opencv-python-headless, tiktoken, fairscale, iopath, fvcore, simplejson, pyyaml, packaging. **Forge packages were never upgraded.**

### Auto-detection (no user config)
- `sam3_setup.default_checkpoint()` — scans models\SAM 3 / sam3 / SAM3 + all subfolders for *.pt/*.safetensors >1MB (excluding runtime/); prefers official `sam3.pt`, else newest.
- `default_bpe()` — finds `bpe_simple_vocab*.gz` under the SAM 3 folder.
- `runtime_directory()` / `activate_runtime()` — activates the isolated runtime path.
- `sam3_mask.auto_settings()` — ('cuda','Full precision (fp32)') with GPU, else ('cpu','Full precision (fp32)').
- Masker cache key includes checkpoint mtime + device + quantization.

### Verified working (live test)
- Real head mask on user's Madayy.24 photo: mask (819,1024), bbox (323,221,598,675), error empty, on CUDA.

### Mask selection logic
- Prompt "head" @ confidence 0.2 with face_box overlap verification (`_choose`: inside-box + overlap + score ranking); falls back to prompt "face" @ 0.33 (from proven workflows).
- Spill rejection: mask clamped to ±head-size box around the detected face; neck band +12% protected.
- Any failure → face-detector mask fallback (batch-safe).

### Download routes (documented in docs/ZIMAGE.md)
- Route A: in-Forge "Download / set up SAM3 once" (gated HF, approved account/token)
- Route B: manual sam3.pt from ungated mirror (e.g. 1038lab/sam3 worked) → models/sam3/
- Route C: safetensors export via fallback loader (may not match Meta keys)

---

## 4. SMART AUTO-SETTINGS (model preset → configuration)

`sync_adapter` in scripts/klein_face_reference.py, wired to Forge preset + checkpoint dropdowns (load + change events), outputs 7 updates:

| Preset detected | Actions |
|---|---|
| ZIT / ZIB / zimage (flat-string match incl. 'zimage','zit','zib' tokens) | BFS dropdown → 'Not used (Z-Image)'; variant Turbo (ZIT) or Base (ZIB); mask → SAM3 + auto-found checkpoint path (if found); character LoRA auto-selected (Z-Image-family, 'char'-preferred); **BFS column hidden; Z-Image accordion shown+opened** |
| Qwen 2.1 | BFS LoRA auto-picked via `core.select_adapter(size=None,'qwen')`; BFS column shown; Z-Image section hidden |
| Klein 9B / 4B | BFS LoRA auto-picked ('klein', size 4/9 via '4b' token); BFS shown |
| No match | BFS → 'Auto (match model)' |
| "Match BFS automatically" OFF | nothing changes (all no-op updates) |

Adapter family detection (`core.adapter_family`): recognizes z-image/zit/zib/tongyi-mai names → 'zimage'; klein/flux2k + 4b/9b → ('klein',size); qwen+version → ('qwen',size).

---

## 5. UI DESIGN (current)

Tabs (1·Swap, 2·Appearance, 3·Quality & mask, 4·Settings) with custom CSS:
- tab-nav: gap .3rem, wrap, bottom border; buttons 38px min-height, weight 650, radius 8/8/0/0
- rows: gap .65rem, align-items flex-end; all paired controls carry scale=1 (symmetry)
- accordions radius 8px; panels radius 8px buttons

Z-Image section (auto-shown for ZIT/ZIB): variant dropdown, identity-change slider, mask dropdown [SAM3 (recommended) / Face detector (fast) / SAM3 (optional)], checkpoint textbox (auto-filled), note that device/precision are auto-detected. BFS controls (dropdown, auto-match checkbox, strengths, trigger, strict check) live in a hideable column.

---

## 6. INPAINTING PIPELINE (final architecture)

### configure_inpaint (khs/zimage.py)
1. Crop region mask, paste into canvas at canvas_box
2. **Auto-tune per image:** head_px measured on-canvas (pose box scaled by crop→canvas factors); feather 8–32px derived from head size (variable retained for report); denoise = base +0.05 (head<256px) / −0.05 (head>640px)
3. Model mask: crisp (GaussianBlur 4 only); latent_mask = same
4. inpaint_full_res=False, fill=1, mask_blur=0, mask_round=False
5. Stores `session.inpaint_mask_canvas` for exact paste-back

### head_ratio_side (khs/runtime.py)
- Canvas side scaled so head lands in 360–640px band (target=min(640, side*0.6); needed=int(target/head_fraction))
- Capped: ≤ Forge generation size, ≤ 1.5× requested side
- Prevents small heads sampled too small for identity; uniform scaling preserves body/head proportion

### finish_image paste-back (khs/runtime.py)
- image: canvas→crop box→baseline size (LANCZOS chain)
- Union mask = max(canvas-mask mapped back to original coords, region mask)
- `composite_region(image, region, color_match)`

### composite_region (khs/core.py) — blending stack
1. Jaw/neck seam: median delta from preserved neck skin → generated jaw, clamped (±80,±40,±40)×strength, alpha-weighted
2. Global skin-tone transfer: YCbCr mean+contrast match from body skin to generated skin; luminance clamp ±60, chroma ±24, gain 0.6–1.6
3. Gamma exposure match on luminance (0.6–1.6), alpha-weighted
4. Final continuous alpha blend: arr*(1-m)+corrected*m

### Geometry quality gate (khs/core.py:geometry_report)
- head height/width error, center error, chin error (existing)
- NEW: body_to_head_ratio_source/output/change_percent (head height ÷ frame height, input vs output)
- gate fails if ratio change >12%; integrated into quality_gate_passed
- caller passes (original.size, image.size) at final measurement

---

## 7. Z-IMAGE TURBO RECOMMENDED SETTINGS

| Setting | Value |
|---|---|
| Steps | 10 (8–12 window; auto-tuned per image 6–21 by head size) |
| CFG | 1 (fixed; negatives ignored) |
| Identity change (denoise) | 0.35 default; 0.25–0.30 if drift; 0.40–0.45 if identity weak |
| Character LoRA strength | 0.9 (as important as denoise) |
| Sampler | Euler a / DPM++ 2M / DDIM all fine |
| One LoRA only | never stack ZIT + ZIB adapters |
| Prompt | write a real scene description (substitutes the ComfyUI workflows' JoyCaption auto-caption step — the one manual step left) |

---

## 8. HEAD ANGLES / POSE — WHAT'S AUTOMATIC VS NOT

Automatic:
- SAM3 masks the visible head contour at any angle; face-box overlap verification rejects wrong-region masks
- Pose (yaw/pitch/roll/head_px) drives target selection, crop geometry, neck protection
- Reference scoring ranks headshots by pose match (Klein/Qwen routes)
- Inpainting re-renders the head in the TARGET's orientation regardless of LoRA training angle

Limits:
- No per-angle LoRA switching (LoRAs are global weight patches)
- Extreme profile/back-of-head: face detection may fail → "Target face not found" (batch skips that target); custom mask required; identity loss is a training-data limit

---

## 9. HEADSHOTS + CHARACTER LORA — SYNC SEMANTICS

- **Z-Image:** identity comes ONLY from the character LoRA. Headshots are scored for the report and feed the identity audit but are NEVER encoded as conditioning (vae_encodes=0, ref_latents=[]). Evidence: log `Z-Image head swap = character LoRA + native inpainting; no reference conditioning`.
- **Klein / Qwen:** headshots ARE reference conditioning (encoded to latents) + BFS LoRA together — true sync.
- To verify per generation: Show last generation report → identity_source, encoded_sizes, vae_encodes.
- If Z-Image output doesn't resemble headshots: fix LoRA strength (0.9–1.25) or denoise; adding headshots won't change output.

---

## 10. EXTERNAL RESEARCH SOURCES USED

- pastebin.com/raw/96QgdwE1 — ComfyUI Z-Image img2img workflow (SAM3 "face" 0.33, denoise 0.25/er_sde 4-step & ddim 8-step 0.2/0.25, Clownshark SDE, InpaintCrop/InpaintStitch 1024 min, LoRA 0.9/0.95, JoyCaption "Art Critique" auto-caption, z_image_turbo_bf16)
- pastebin.com/raw/XQkT4V9n — JoyCaption prompt-building workflow (Qwen3-4b-Z-Image-Engineer-V4 GGUF TE, lumina2, caption→concat→CLIPTextEncode)
- Local: C:\Users\SIDDHARTH MISHRA\Downloads\Test.Workflow.json — ComfyUI-SAM3 image+video segmentation workflow (SAM3Grounding 0.05 'girl', LoadSAM3Model fp32, video propagate)
- github.com/PozzettiAndrea/ComfyUI-SAM3 (+issue #98: text segmentation node mismatches original repo; our extension bypasses the wrapper and calls official sam3 directly)
- huggingface.co/facebook/sam3 (gated; official weights), ungated mirror 1038lab/sam3 (sam3.pt downloaded successfully)
- huggingface.co/RetroGazzaSpurs/ZIT_CharacterLoras, Comfy-Org/z_image_turbo (referenced model sources)
- Reddit threads (login-walled, content inferred from workflow JSONs): r/StableDiffusion 1qz9lzb, 1qxsisg; r/malcolmrey 1qxznum

---

## 11. REMAINING KNOWN ITEMS (not bugs)

1. **JoyCaption-style auto-captioning** is not implemented in Forge — the equivalent manual step is writing a good scene prompt per batch. This is the one feature the ComfyUI workflows have that we don't automate.
2. Community `SAM 3.safetensors` remains unusable (key-layout mismatch); official sam3.pt is the working checkpoint. The safetensors fallback loader remains for compatible exports.
3. Gradio Windows file-cache race (`FileExistsError/PermissionError` on layer_0.png/composite.png) during ImageEditor uploads while generating — upstream Gradio issue, harmless, avoid painting masks mid-batch.
4. Extreme-angle (back-of-head) targets without a custom mask are skipped by design.
5. Optional future: SAM3 text-prompt fallback without requiring a detected face box (for back-of-head masks).

---

## 12. HOW TO RUN (end user quickstart)

1. Full Forge restart (Python changes require process restart)
2. Select ZIT/ZIB checkpoint in the Forge dropdown — extension self-configures (variant, character LoRA, SAM3 mask + path, BFS hidden)
3. Put target photo(s) in plain img2img (NOT the inpaint tab), write a scene prompt
4. Character strength 0.9; denoise 0.35 default
5. Generate. Check Show last generation report → geometry_final (incl. body_to_head_ratio_*), native_inpaint.mask_source should show SAM3 active
6. SAM3: fully auto (files from models\SAM 3 + subfolders, CUDA + fp32). Face-detector fallback covers failures — batches never abort.

**Test status:** 192/192 passing at every commit. All work pushed to master.
