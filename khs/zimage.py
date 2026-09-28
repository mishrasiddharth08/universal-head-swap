"""Native Z-Image inpainting helpers; no reference encoder or diffusion model load."""
from pathlib import Path
import threading

import numpy as np
from PIL import Image,ImageFilter

from . import core

_MASK_LOCK = threading.RLock()
_MASKER = None
_MASKER_KEY = None


def clear_mask_cache():
    global _MASKER, _MASKER_KEY
    with _MASK_LOCK:
        if _MASKER is not None:
            _MASKER.release(clear_cache=True)
        _MASKER = None
        _MASKER_KEY = None


def has_custom_mask(value):
    if isinstance(value, dict):
        value = value.get('composite') if value.get('composite') is not None else value.get('background')
    if value is None:
        return False
    if not isinstance(value, Image.Image):
        value = core.rgb_image(value)
    if value.mode in ('RGBA', 'LA'):
        rgba = value.convert('RGBA')
        visible = Image.new('RGB', value.size)
        visible.paste(rgba, mask=rgba.getchannel('A'))
        value = visible
    return value.convert('L').getbbox() is not None


def prepare_mask(image, pose, cfg, custom_mask=None):
    """Use an explicit mask first; SAM3 stays optional and runs on CPU only."""
    global _MASKER, _MASKER_KEY
    if has_custom_mask(custom_mask):
        return custom_mask
    if not str(cfg.get('zimage_mask_source') or '').startswith('SAM3'):
        return None
    if not pose:
        # No detected face: without a mask source there is nothing to segment, so
        # fall back to the face-detector path instead of aborting folder batches.
        print('[UniversalHeadSwap] SAM3 head selection needs a detected face; using the face detector mask for this target.')
        return None
    typed=Path(str(cfg.get('sam3_checkpoint') or '').strip()).expanduser()
    if not typed.is_file():
        from .sam3_setup import default_checkpoint
        found=default_checkpoint()
        if not found:
            print('[UniversalHeadSwap] No SAM3 checkpoint found (models/SAM 3); using the face detector mask.')
            return None
        checkpoint=found
    else:
        checkpoint=typed
    from .sam3_mask import SAM3Masker
    stat = checkpoint.stat()
    device = 'cuda' if cfg.get('sam3_device') == 'GPU (CUDA)' else 'cpu'
    quant = cfg.get('sam3_quantization') or 'Full precision (fp32)'
    key = (str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns, device, quant)
    with _MASK_LOCK:
        if key != _MASKER_KEY:
            clear_mask_cache()
            _MASKER = SAM3Masker(checkpoint, device=device, quantization=quant)
            _MASKER_KEY = key
        sample = image.copy()
        sample.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        face = core.scale_pose(pose, image.size, sample.size)['box']
        try:
            mask = _MASKER.mask(sample, prompt='head', face_box=face)
            if mask is None or mask.getbbox() is None:
                # Missing mask on one photo must not kill a folder batch; fall back to the detector mask.
                print('[UniversalHeadSwap] SAM3 found no reliable head mask (' + (_MASKER.error or 'no matching head') + '); using the face detector mask for this target.')
                return None
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            # Reject segmentation spill into a neighbouring person/body.
            x0, y0, x1, y1 = pose['box']; w=x1-x0; h=y1-y0
            arr=np.asarray(mask).copy()
            left=max(0,int(x0-w)); right=min(image.width,int(x1+w))
            top=max(0,int(y0-h)); bottom=min(image.height,int(y1+h*.12))
            arr[:top]=0; arr[bottom:]=0; arr[:,:left]=0; arr[:,right:]=0
            result=Image.fromarray(arr,'L')
            if result.getbbox() is None:
                raise ValueError('SAM3 mask missed the selected head. Use a custom mask.')
            return result
        finally:
            # Keep only the small CPU mask cache between previews/generations.
            _MASKER.release()


def protect_sam3_neck(region,pose,cfg):
    """Clamp automatic SAM3 feathering after blur; explicit masks stay user-controlled."""
    if not str(cfg.get('zimage_mask_source') or '').startswith('SAM3') or has_custom_mask(cfg.get('custom_mask')) or not pose:
        return region
    _,y0,_,y1=pose['box']; height=y1-y0
    arr=np.asarray(region.mask,dtype=np.float32).copy()
    top=max(0,min(arr.shape[0],round(y1-height*.02)))
    bottom=max(top+1,min(arr.shape[0],round(y1+height*.12)))
    if top<arr.shape[0]:
        bottom=min(bottom,arr.shape[0])
        arr[top:bottom]*=np.linspace(1,0,bottom-top,dtype=np.float32)[:,None]
        arr[bottom:]=0
    region.mask=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),'L')
    return region


def configure_inpaint(session):
    region=session.region
    mask=region.mask.crop(region.box)
    x0,y0,x1,y1=session.canvas_box
    cw,ch=session.canvas_size
    canvas=Image.new('L',(cw,ch),0)
    canvas.paste(mask.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS),(x0,y0))
    if canvas.getbbox() is None:
        raise ValueError('Z-Image edit mask is empty.')
    # Auto-adjust to the image input: the feather and denoise follow the head's
    # on-canvas pixel size, so a small face in a wide crop and a full-frame
    # portrait both get appropriate edges and identity strength instead of
    # one fixed setting.
    base=float(session.cfg['zimage_denoise'])
    denoise=base; head_px=0
    pose=getattr(session,'target_pose',None)
    if pose and region.box:
        rx0,ry0,rx1,ry1=region.box
        sx=(x1-x0)/max(1,rx1-rx0); sy=(y1-y0)/max(1,ry1-ry0)
        head_px=max((pose['box'][2]-pose['box'][0])*sx,(pose['box'][3]-pose['box'][1])*sy)
    if head_px>=16:
        feather=int(round(min(32,max(8,head_px*0.06))))
        if head_px<256: denoise=min(1.0,base+0.05)      # tiny head: re-render more
        elif head_px>640: denoise=max(0.1,base-0.05)    # large head: keep identity
    else:
        feather=max(8,min(32,cw//48))
    # Soft edges let Turbo/Base re-render cleanly into the surrounding skin and hair
    # instead of stopping on a hard circle. The latent mask stays slightly tighter
    # than the overlay mask so the outer feather is painted from well-denoised
    # context rather than a half-strength latent.
    soft=canvas.filter(ImageFilter.GaussianBlur(feather))
    latent=soft.filter(ImageFilter.GaussianBlur(max(4,feather//2)))
    settings=dict(image_mask=soft, latent_mask=latent,
                  inpaint_full_res=False, inpainting_mask_invert=0,
                  inpainting_fill=1, mask_blur=0, mask_round=False,
                  denoising_strength=denoise)
    for key,value in settings.items():
        session.set_p(key,value)
    for key in ('overlay_images','mask_for_overlay','paste_to','mask','nmask'):
        session.set_p(key,None)
    session.host.dynamic.is_referencing=False
    if hasattr(session.host.dynamic,'edit'):
        session.host.dynamic.edit=False
    session.analysis['native_inpaint']={'denoise':denoise,'base_denoise':base,
        'mask_source':session.cfg.get('zimage_mask_source','Face detector (fast)'),
        'mask_feather_px':feather,'head_px':round(head_px) if head_px else None,
        'sampling_size':[cw,ch],'reference_encodes':0}


def prepare_batch(session,index):
    p=session.p; h=session.host
    session.model.ref_latents=[]; session.model.ini_latent=None
    h.dynamic.ref_latents=[]; h.dynamic.is_referencing=False
    if session.refs:
        session.selected_ref,_=core.select_reference(session.scored,session.cfg,index,len(session.refs))
    p.clear_prompt_cache()
    p.extra_generation_params['Z-Image head swap']='character LoRA + native inpainting; no reference conditioning'
    p.extra_generation_params['Z-Image mode']=session.cfg['zimage_variant']
    p.extra_generation_params['Z-Image denoise']=session.cfg['zimage_denoise']
    session.owner.last_report={'version':core.VERSION,'model_family':'zimage','image':index+1,
        'character_lora':session.char,'identity_source':'character LoRA',
        'analysis':session.analysis,'plan':session.plans[index].report(),
        'crop_box':session.region.box,'encoded_sizes':[], 'vae_encodes':0,
        'sampling_size':list(session.canvas_size),'status':'Generating'}
    h.shared.state.textinfo='Z-Image: character LoRA + protected head inpainting'


def suppress_native_overlay(p, overlay):
    """The extension applies the final mask once, after restoring the original canvas."""
    session=getattr(p,'_khs_session',None)
    if (session is None or session.p is not p or session.family!='zimage'
            or session.region is None or getattr(p,'_ad_inner',False)):
        return
    overlay.overlay_image=None
    overlay.mask_for_overlay=None
