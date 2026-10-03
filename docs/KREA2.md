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
5. Keep automatic instructions enabled; the extension adds `head_swap: replace the head with the reference head.` once. Add scene or appearance requests in the normal prompt.
6. Press Forge's normal **Generate** button.

The source is image 1 and the reference head is image 2. BFS is model-only at strength 1. The upstream simple workflow uses Euler, Simple, 8 steps, CFG 1 and denoise 1. Its raw-model note recommends about 40 steps and CFG 3-4. Automatic SAM3 head masks apply to Z-Image; the separate spot-cleanup detector can still review marks in completed photos. Review every result at 100% zoom.

Automated tests check routing, adapter matching and contracts. GPU execution and visual quality remain unverified until recorded in `VALIDATION.md`.
