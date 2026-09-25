# Automatic fine-grid cleanup

Enable **More options → Detail → Fine-grid cleanup → Automatically reduce fine 2-pixel grids**. Default: off; strength: 0.5. Images without a detected phase-locked grid pass through unchanged. A clean decision is a heuristic, not proof that all artifacts are absent. Strength controls the blend, not the detection threshold. Compare at 100% zoom; patterned fabrics and other real repeating texture can resemble a grid.

## Source and adaptation

Adapted from [ComfyUI-DeGrid by lunaaispace-eng](https://github.com/lunaaispace-eng/ComfyUI-DeGrid), pinned commit `5699bc33f71e1be12523fdea105cc9c2abfe1cd0`. The original source is Apache-2.0: see [license](DEGRID_LICENSE.txt) and [source record](DEGRID_SOURCE.json). Changes: a Pillow/NumPy CPU adaptation, row-tiled filtering, adjustable final blend, protected-edit integration and metadata reporting. No ComfyUI install, shader runtime, GPU model download or extra dependency is needed.

Original user references: [discussion](https://www.reddit.com/r/StableDiffusion/comments/1wlv3tf/qwen_image_21_noisepatterningmoire_mild_workaround/) and [shader](https://pastebin.com/v7y1z0SH). The integrated automatic algorithm follows the GitHub implementation rather than assuming the earlier shader is identical.

## Behavior

The automatic path uses the nine-tap bandpass kernel `[1,-8,28,-56,70,-56,28,-8,1]/256`, subtracting horizontal and vertical Nyquist components and adding back their intersection. Phase-folding measures the repeating lattice; clean images bypass filtering. The correction is capped using measured image content rather than applying an unrestricted notch. Float intermediate values are clipped/rounded once at output. Original alpha is preserved by the standalone filter.

The pipeline runs on the decoded image at the start of Klein finishing, before Klein resizes, sharpens or composites it. This avoids amplifying the grid before removal. Existing protected-edit compositing then restores pixels outside the edit mask; existing optional full-frame finishing effects retain their behavior. Final quality checks measure the finished image. External extensions that process pixels before Klein may already have altered the original grid; this cannot undo arbitrary upstream processing.

This targets repeating 2-pixel decoder grids, not general moiré, all noise, inpainting boundaries, tattoos or piercings. Fine texture can soften when it is indistinguishable from the pattern. Detection thresholds are not calibrated to every Klein output. The original repository discusses Qwen and Wan decoder artifacts; this integration does not add Qwen generation support. No universal quality guarantee is implied.

The two settings are appended to preserve older argument positions and are included in presets and generation metadata. The original seven-tap helper remains available internally for regression/reference comparison; the UI uses the automatic nine-tap path. See VALIDATION.md for actual test evidence and remaining visual acceptance.
