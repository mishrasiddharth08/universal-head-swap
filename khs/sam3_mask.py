"""Optional, local-only SAM3 text-mask adapter.

Nothing is imported, installed, or downloaded until ``mask`` is called.
Call ``release`` before the generation model is loaded to return SAM3 VRAM.
"""
from contextlib import nullcontext, suppress
import gc
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from .core import BoundedCache, image_hash


QUANTIZATIONS = ('Auto (detected)', 'Full precision (fp32)', 'Half (fp16)', 'BFloat16', 'FP8 E4M3 (GPU)',
                 'FP8 E5M2 (GPU)', 'Dynamic INT8 (CPU only)', 'INT8 weight-only',
                 'INT4 weight-only')


def auto_settings():
    """Auto-detect the best SAM3 device and precision for this machine.

    SAM3's processor feeds Float inputs regardless of weight dtype, so a
    global BFloat16/FP16 cast breaks inference with a dtype mismatch. The
    reliable automatic choice is therefore fp32 on whichever device exists.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda', 'Full precision (fp32)'
    except Exception:
        pass
    return 'cpu', 'Full precision (fp32)'


class SAM3Masker:
    def __init__(self, checkpoint, bpe_path=None, device="cuda", cache_bytes=8 * 1024**2,
                 loader=None, quantization='Full precision (fp32)'):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.bpe_path = Path(bpe_path) if bpe_path else None
        self.device = device
        self.quantization = quantization if quantization in QUANTIZATIONS else 'Full precision (fp32)'
        self.loader = loader
        self.processor = None
        self.model = None
        self.error = ""
        self.lock = threading.RLock()
        self.cache = BoundedCache(cache_bytes)

    @property
    def available(self):
        return bool(self.loader or (self.checkpoint and self.checkpoint.is_file()))

    def _load(self):
        if self.processor is not None:
            return
        if not self.available:
            raise RuntimeError("SAM3 checkpoint is missing; automatic downloads are disabled.")
        if self.loader:
            loaded = self.loader()
            self.model, self.processor = loaded if isinstance(loaded, tuple) else (None, loaded)
            if self.model is not None and hasattr(self.model, "eval"):
                self.model.eval()
            return
        from .sam3_setup import activate_runtime
        activate_runtime()
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError("SAM3 is not installed in this Forge environment.") from exc
        device = self.device
        quant = self.quantization
        if quant in ('Dynamic INT8 (CPU only)', 'INT8 weight-only', 'INT4 weight-only') and not str(device).startswith('cpu'):
            device = 'cpu'  # torchao weight quantization runs on CPU here
        kwargs = dict(checkpoint_path=str(self.checkpoint), load_from_HF=False,
                      device=device, compile=False)
        if self.bpe_path:
            if not self.bpe_path.is_file():
                raise RuntimeError("SAM3 tokenizer file is missing.")
            kwargs["bpe_path"] = str(self.bpe_path)
        self.model = None
        safetensors_state = None
        try:
            self.model = build_sam3_image_model(**kwargs)
        except Exception as first_error:
            # The official loader expects a pickled .pt. Community checkpoints are
            # often safetensors; build unweighted and load the state dict directly.
            self.model = None
            try:
                from safetensors.torch import load_file
                unweighted = dict(kwargs); unweighted['checkpoint_path'] = None
                self.model = build_sam3_image_model(**unweighted)
                safetensors_state = load_file(str(self.checkpoint))
                missing, unexpected = self.model.load_state_dict(safetensors_state, strict=False)
                if len(missing) > len(self.model.state_dict()) * 0.5:
                    raise RuntimeError(f'checkpoint keys do not match SAM3 ({len(missing)} missing)')
                print(f'[UniversalHeadSwap] SAM3 loaded from safetensors: {len(safetensors_state)} tensors'
                      + (f'; {len(missing)} missing' if missing else ''))
            except Exception as exc:
                self.model = None
                raise RuntimeError(
                    f'SAM3 checkpoint could not be loaded ({type(first_error).__name__}: '
                    + str(first_error)[:120] + '; safetensors fallback: ' + str(exc)[:120] + ')') from exc
        self._apply_quantization(quant, device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.processor = Sam3Processor(self.model, device=device)

    def _apply_quantization(self, quantization, device):
        """Cast or quantize the loaded model in place.

        Community checkpoints often store BFloat16 weights while the CPU
        processor feeds Float inputs, which crashes with a dtype mismatch.
        On CPU the model is therefore always normalized to a single dtype.
        """
        import torch
        cast=getattr(self.model,'to',None)
        if str(device).startswith('cpu'):
            if quantization in ('Dynamic INT8 (CPU only)', 'INT8 weight-only', 'INT4 weight-only'):
                # torchao quantizers manage their own weight dtypes.
                pass
            elif cast:
                cast(torch.float32)
            return
        if quantization in ('Dynamic INT8 (CPU only)', 'INT8 weight-only', 'INT4 weight-only'):
            return
        if quantization in ('Half (fp16)', 'BFloat16', 'FP8 E4M3 (GPU)', 'FP8 E5M2 (GPU)'):
            # SAM3's processor feeds Float inputs regardless of weight dtype; a
            # global half/float8 cast always crashes inference with a dtype
            # mismatch (including values persisted from older saved settings).
            print('[UniversalHeadSwap] SAM3 ' + quantization + ' is not compatible with its processor; running full precision.')
        # The checkpoint stores BFloat16 weights while the processor feeds Float
        # inputs, so GPU inference also requires a fp32-normalized model.
        if cast:
            cast(torch.float32)
        if quantization == 'Dynamic INT8 (CPU only)':
            try:
                from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight_quant
            except ImportError:
                try:
                    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
                    quantize_(self.model, int8_dynamic_activation_int8_weight())
                    return
                except ImportError as exc:
                    raise RuntimeError('Dynamic INT8 needs the torchao package on CPU: ' + str(exc)) from exc
            quantize_(self.model, int8_dynamic_activation_int8_weight_quant())
        elif quantization == 'INT8 weight-only':
            from torchao.quantization import quantize_
            try:
                from torchao.quantization import int8_weight_only_quant
                quantize_(self.model, int8_weight_only_quant())
            except ImportError:
                from torchao.quantization import int8_weight_only
                quantize_(self.model, int8_weight_only())
        elif quantization == 'INT4 weight-only':
            from torchao.quantization import quantize_
            try:
                from torchao.quantization import int4_weight_only_quant
                quantize_(self.model, int4_weight_only_quant())
            except ImportError:
                from torchao.quantization import int4_weight_only
                quantize_(self.model, int4_weight_only())

    @staticmethod
    def _inference_context():
        try:
            import torch
            return torch.inference_mode()
        except ImportError:
            return nullcontext()

    @staticmethod
    def _cpu(value):
        for name in ("detach", "float", "cpu"):
            method = getattr(value, name, None)
            if method:
                value = method()
        return np.asarray(value)

    @staticmethod
    def _choose(masks, boxes, scores, face_box):
        count = len(masks)
        if not count:
            return None
        scores = np.ravel(scores) if scores is not None else np.zeros(count)
        if face_box is None:
            return int(np.argmax(scores[:count]))
        if boxes is None or len(boxes) != count:
            return None
        fx0, fy0, fx1, fy1 = map(float, face_box)
        fc = ((fx0 + fx1) / 2, (fy0 + fy1) / 2)
        best = None
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = map(float, box)
            ix = max(0, min(fx1, x1) - max(fx0, x0))
            iy = max(0, min(fy1, y1) - max(fy0, y0))
            overlap = ix * iy / max(1, (fx1 - fx0) * (fy1 - fy0))
            inside = x0 <= fc[0] <= x1 and y0 <= fc[1] <= y1
            rank = (inside, overlap, float(scores[i]) if i < len(scores) else 0)
            if best is None or rank > best[0]:
                best = (rank, i)
        return best[1] if best and best[0][1] > 0 else None

    def mask(self, image, prompt="head", confidence=0.2, face_box=None):
        image = image.convert("RGB")
        key = (image_hash(image), prompt.strip().lower(), round(float(confidence), 3),
               tuple(round(float(x), 1) for x in face_box) if face_box else None)
        with self.lock:
            self.error = ""
            cached = self.cache.get(key)
            if cached is not None:
                return cached.copy()
            try:
                self._load()
                with self._inference_context():
                    if hasattr(self.processor, "set_confidence_threshold"):
                        self.processor.set_confidence_threshold(float(confidence))
                    state = self.processor.set_image(image)
                    output = self.processor.set_text_prompt(prompt.strip(), state)
                masks = self._cpu(output.get("masks", []))
                if masks.ndim == 4:
                    masks = masks[:, 0]
                boxes = self._cpu(output.get("boxes")) if output.get("boxes") is not None else None
                scores = self._cpu(output.get("scores")) if output.get("scores") is not None else None
                chosen = self._choose(masks, boxes, scores, face_box)
                if chosen is None:
                    return None
                values = np.asarray(masks[chosen])
                if values.dtype == np.bool_:
                    binary = values
                else:
                    finite = values[np.isfinite(values)]
                    threshold = 0.5 if finite.size and finite.min() >= 0 and finite.max() <= 1 else 0.0
                    binary = values > threshold
                result = Image.fromarray(binary.astype(np.uint8) * 255, "L")
                if result.size != image.size:
                    result = result.resize(image.size, Image.Resampling.NEAREST)
                self.cache.put(key, result.copy(), result.width * result.height)
                return result
            except Exception as exc:
                self.error = str(exc)
                self.release()
                return None

    def candidates(self, image, prompts, confidence=0.4, limit=24):
        """Return bounded candidate masks; caller must obtain a user's selection."""
        image=image.convert('RGB'); image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        results=[]
        with self.lock:
            self.error=''
            try:
                self._load()
                with self._inference_context():
                    self.processor.set_confidence_threshold(float(confidence))
                    state=self.processor.set_image(image)
                    for prompt in list(dict.fromkeys(prompts))[:12]:
                        if hasattr(self.processor,'reset_all_prompts'): self.processor.reset_all_prompts(state)
                        output=self.processor.set_text_prompt(str(prompt),state)
                        masks=self._cpu(output.get('masks',[]))
                        if masks.ndim==4: masks=masks[:,0]
                        scores=self._cpu(output.get('scores',[])).reshape(-1)
                        for i,values in enumerate(masks):
                            if i>=len(scores) or not np.isfinite(scores[i]) or scores[i]<confidence: continue
                            finite=values[np.isfinite(values)]
                            threshold=.5 if finite.size and finite.min()>=0 and finite.max()<=1 else 0
                            mask=Image.fromarray((values>threshold).astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST)
                            if mask.getbbox() is None: continue
                            # Suppress duplicate proposals returned by related category prompts.
                            binary=np.asarray(mask)>0
                            duplicate=False
                            for prior in results:
                                previous=np.asarray(prior['mask'])>0
                                union=np.count_nonzero(binary|previous)
                                if union and np.count_nonzero(binary&previous)/union>.8: duplicate=True;break
                            if duplicate: continue
                            results.append({'label':str(prompt),'score':float(scores[i]),'mask':mask})
                            if len(results)>=limit: return results
                return results
            except Exception as exc:
                self.error=str(exc)
                raise RuntimeError('Automatic spot detection unavailable: '+self.error) from exc
            finally:
                self.release()

    def release(self, clear_cache=False):
        """Unload SAM3; cached masks remain CPU-only unless explicitly cleared."""
        with self.lock:
            self.processor = None
            self.model = None
            if clear_cache:
                self.cache.clear()
            gc.collect()
            with suppress(Exception):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            with suppress(Exception):
                from backend import memory_management
                memory_management.soft_empty_cache()

    close = release
