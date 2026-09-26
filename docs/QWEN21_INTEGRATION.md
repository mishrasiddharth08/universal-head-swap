# Project Invisible Qwen 2.1 companion requirement

The dedicated Qwen engine intercepts generation before Forge's standard script callbacks. Head Swap alone cannot make an older copy of that engine consume its controls.

Update both Universal Head Swap and the Project Invisible Qwen 2.1 extension, then restart Forge completely. Browser refresh is insufficient. The Qwen companion fixes are maintained separately; they are not installed by this repository.

## Integration contract

- The companion opens the always-on script's `headswap_external_context(p)` context around generation. Disabled Head Swap yields no session.
- For each output it calls `session.prepare(index, prompt, negative, seed, cfg_scale, turbo)` and uses the returned plan and reference pair. The plan includes the selected BFS and character LoRAs; Turbo remains active alongside them.
- The companion calls `session.finish_image(image, index)` before saving, records `session.metadata()`, honors seed lock, and verifies the saved output path.
- The context always closes, restoring request state even when generation fails.

Use the generation report to verify the selected reference and adapters. Absence of a Head Swap reference message means the companion is not invoking the bridge; repeated generation will not fix that integration mismatch.

## Validation scope

130 tests cover this repository. The separately updated companion passed 132 tests. A real RTX 5090 test used Qwen INT8 ConvRot, 13 headshots, BFS, a character LoRA and Viggle v0.2.1 Turbo. The protected result saved successfully; face identity and face-height/center checks passed, with advisory width/reference-consistency warnings. Other quantizations and real Klein generation remain unverified.

## Companion follow-up fixes

The companion also needs corrected Turbo control updates (no nested Gradio skip dictionaries), numeric speed-strength recovery, and CFG support with empty negative text. These companion changes are maintained separately and are not shipped by this repository. Qwen receives native `<image1>`/`<image2>` references and plain instructions. The dedicated engine currently uses Euler, irrespective of Forge sampler selection.
