"""Normalize white-edit/black-preserve masks without treating photos as masks."""
import numpy as np
from PIL import Image,ImageOps
from pathlib import Path
import base64,io

def editor_mask(value,load_image):
    editor_value=isinstance(value,dict)
    def image(item):
        if item is None:return None
        if isinstance(item,dict):
            item=next((item.get(k) for k in ('image','path','name','composite','background') if item.get(k) is not None),None)
        if isinstance(item,(str,Path)):
            if str(item).startswith('data:image/'):
                source=io.BytesIO(base64.b64decode(str(item).split(',',1)[1]))
            else:source=item
            with Image.open(source) as opened:item=opened.copy()
        elif isinstance(item,np.ndarray):item=Image.fromarray(item)
        elif not isinstance(item,Image.Image):item=load_image(item)
        return ImageOps.exif_transpose(item)
    def flattened(item):
        if item.mode in ('RGBA','LA') or 'transparency' in item.info:
            rgba=item.convert('RGBA');base=Image.new('RGB',rgba.size,'black');base.paste(rgba,mask=rgba.getchannel('A'));item=base
        return item.convert('L')
    if isinstance(value,dict):
        background=image(value.get('background'))
        layers=value.get('layers') or []
        if background is not None:
            rgb=np.asarray(background.convert('RGB'))
            photo_backed=bool(np.any(rgb[...,0]!=rgb[...,1]) or np.any(rgb[...,1]!=rgb[...,2]))
            # A grayscale portrait is still a photo. Only binary canvases
            # are unambiguous editable mask backgrounds. Soft masks enter
            # through the mask upload control or as direct API images.
            if photo_backed or any(flattened(background).histogram()[1:255]):
                if not layers:raise ValueError('A target photo is not an edit mask. Paint white edit strokes, or create a detected-head mask.')
                canvas=Image.new('RGBA',background.size,(0,0,0,0))
                for layer in layers:
                    layer=image(layer).convert('RGBA')
                    if layer.size!=canvas.size:raise ValueError('Paint layer coordinates do not match the mask background')
                    canvas=Image.alpha_composite(canvas,layer)
                result=flattened(canvas)
                return result
        value=value.get('composite') if value.get('composite') is not None else value.get('background')
    if value is None:return None
    result=flattened(image(value))
    return None if editor_value and not result.getbbox() else result

def resize_mask(mask,size):
    # Lanczos can create nonzero ringing outside a hard painted area.
    binary=not any(mask.histogram()[1:255])
    return mask.resize(size,Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR)
