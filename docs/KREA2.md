# Krea2 + BFS v1.1

Universal Head Swap detects Forge presets named `krea` or `krea2` and routes them to the native Forge `Krea2` engine/config. It automatically chooses `bfs_head_swap_v1.1_krea2.safetensors`, including when the LoRA is inside a subfolder. FLUX.1 Krea Dev is a different model family and is never selected for this route.

## Files

- Krea2 checkpoint in Forge's normal checkpoint folder
- `qwen3vl_4b` text encoder configured by the Krea2 engine
- Qwen Image VAE configured by the Krea2 engine
- `bfs_head_swap_v1.1_krea2.safetensors` in `models/Lora` or any subfolder
- Optional Krea2-compatible character LoRA

The extension does not redistribute model weights. Upstream: [Krea2 instructions](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/krea-2.md), [simple workflow](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/workflows/Head%20Swap%20Krea%202%20-%20V1%20Simple%20Workflow.json), and [BFS v1.1 weight](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/bfs_head_swap_v1.1_krea2.safetensors).

## Use

1. Load the native Forge Krea2 preset/checkpoint.
2. Open img2img and upload the source/body image.
3. Enable Universal Head Swap and add the reference headshot.
4. Keep **Match BFS to selected model automatically** enabled.
5. Optionally choose a Krea2-family Character LoRA. It is separate from the required BFS head-swap LoRA.
6. Keep automatic instructions enabled; the extension adds `head_swap: replace the head with the reference head.` once. Add scene or appearance requests in the normal prompt.
7. Press Forge's normal **Generate** button.

The source is image 1 and the reference head is image 2. BFS is model-only at strength 1. The upstream simple workflow uses Euler, Simple, 8 steps, CFG 1 and denoise 1. Its raw-model note recommends about 40 steps and CFG 3-4. Native Forge Spectrum remains available and can change the effective step choice and quality; control it in Forge. Automatic SAM3 head masks apply to Z-Image, not Krea2. Review every result at 100% zoom.

## Local validation

- Native BF16 and INT8 DiT generation completed with the BF16 Krea2 text encoder, Qwen VAE, BFS and a Krea2 character LoRA. Both tested BFS versions loaded all 256 character-adapter weights with zero skipped.
- A BF16 batch initially stalled before step 0 near full VRAM. The Krea-only retry completed after reserving working memory. During Krea sessions the extension requests 32% of total VRAM, capped at 12 GiB, preserves any larger user reserve and restores the exact previous value afterward.
- Five different final INT8 + Spectrum images used BFS v1.1 and character strength 0.6. All saved at 1024×1280. The first request took 28.31 seconds; the four warm requests took 18.93–20.54 seconds.
- Protected compositing changed zero pixels outside each head mask.
- Some images still had imperfect face width/height, skin tone, expression, neck appearance or likeness. Spectrum can affect quality, and these timings are not a controlled Spectrum speed comparison.
- Testing used one RTX 5090. It does not establish every quantization, GPU, VRAM size or perfect results. See [`VALIDATION.md`](../VALIDATION.md) for the full evidence.
