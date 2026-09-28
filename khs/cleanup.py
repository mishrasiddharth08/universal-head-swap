"""Local CPU repair of explicitly painted small marks; no automatic body detector."""
import numpy as np
from PIL import Image


def painted_mask(value,size):
    if value is None: return None
    if isinstance(value,dict):
        # Never treat the editor photo/background as a removal mask.
        layers=value.get('layers') or []
        mask=Image.new('L',size)
        for layer in layers:
            if not isinstance(layer,Image.Image):
                from .core import rgb_image
                layer=rgb_image(layer)
            if layer.size!=size: raise ValueError('Spot cleanup: paint on a photo with the same dimensions as the target.')
            rgba=np.asarray(layer.convert('RGBA'))
            selected=(rgba[...,:3].min(axis=2)>127)&(rgba[...,3]>127)
            mask=Image.fromarray(np.maximum(np.asarray(mask),selected.astype(np.uint8)*255))
    elif isinstance(value,Image.Image):
        if value.size!=size: raise ValueError('Spot cleanup mask must match target dimensions.')
        mask=value.convert('L')
    else:
        from .core import rgb_image
        return painted_mask(rgb_image(value),size)
    return mask if mask.getbbox() else None


def repair(image,value):
    mask=painted_mask(value,image.size)
    if mask is None: return image,{'applied':False}
    binary=(np.asarray(mask)>127).astype(np.uint8)*255
    count=int(np.count_nonzero(binary))
    if count==0: return image,{'applied':False}
    if count>image.width*image.height*.03:
        raise ValueError('Spot cleanup: paint only the small marks or jewelry, not broad skin areas (maximum 3% of image).')
    import cv2
    # Work on a bounded crop and paste only marked pixels: all other pixels are exact.
    ys,xs=np.where(binary);pad=24
    box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(image.width,int(xs.max())+pad+1),min(image.height,int(ys.max())+pad+1))
    x0,y0,x1,y1=box
    source=np.asarray(image.convert('RGB')).copy()
    crop=source[y0:y1,x0:x1].copy();local=binary[y0:y1,x0:x1]
    healed=cv2.inpaint(crop,local,3,cv2.INPAINT_TELEA)
    crop[local>0]=healed[local>0];source[y0:y1,x0:x1]=crop
    return Image.fromarray(source),{'applied':True,'marked_pixels':count,'method':'CPU spot repair','scope':'painted pixels only','note':'Inspect repaired texture; large marks and accessories crossing edges may need manual retouching.'}


DETECTION_PROMPTS={
    'Tattoos':['tattoo'],
    'Body piercings':['body piercing','navel piercing','nose piercing','earring'],
    'Cross symbols':['cross symbol'],
    'Other religious symbols':['religious symbol'],
}


def editor_photo(value):
    from .core import rgb_image
    photo=value.get('background') if isinstance(value,dict) else value
    if photo is None: raise ValueError('Upload the target photo in Paint spots to remove first.')
    return rgb_image(photo)


def detect(value,categories,checkpoint,confidence=.4,extra=''):
    from .sam3_mask import SAM3Masker
    from .core import image_hash
    photo=editor_photo(value)
    prompts=[p for category in (categories or []) for p in DETECTION_PROMPTS.get(category,[])]
    prompts.extend(p.strip() for p in str(extra).split(',') if p.strip())
    if not prompts: raise ValueError('Select at least one detection category.')
    proposals=SAM3Masker(checkpoint,device='cpu').candidates(photo,prompts,confidence)
    labels=[f'{i+1}: {p["label"]} ({p["score"]:.0%})' for i,p in enumerate(proposals)]
    preview=photo.copy();preview.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    from PIL import ImageDraw
    drawing=ImageDraw.Draw(preview)
    for i,item in enumerate(proposals):
        mask=item['mask'].resize(preview.size,Image.Resampling.NEAREST)
        box=mask.getbbox()
        if box:
            drawing.rectangle(box,outline='red',width=2);drawing.text(box[:2],str(i+1),fill='red')
    return {'hash':image_hash(photo),'proposals':proposals,'labels':labels},labels,preview


def select_candidates(value,state,selected):
    from .core import image_hash
    photo=editor_photo(value)
    if not state or image_hash(photo)!=state.get('hash'):
        raise ValueError('The target photo changed. Detect again before selecting spots.')
    small=Image.new('L',state['proposals'][0]['mask'].size) if state['proposals'] else Image.new('L',(1,1))
    for label in selected or []:
        if label not in state['labels']: raise ValueError('Detection selection is stale. Detect again.')
        mask=state['proposals'][state['labels'].index(label)]['mask']
        small=Image.fromarray(np.maximum(np.asarray(small),np.asarray(mask)))
    mask=small.resize(photo.size,Image.Resampling.NEAREST)
    layer=Image.new('RGBA',photo.size,'white');layer.putalpha(mask)
    composite=photo.convert('RGBA');composite.alpha_composite(layer)
    return {'background':photo.convert('RGBA'),'layers':[layer],'composite':composite}


def validate_target(value,target):
    mask=painted_mask(value,target.size)
    if mask is None: return
    if np.count_nonzero(np.asarray(mask)>127)>target.width*target.height*.03:
        raise ValueError('Spot cleanup: paint only small marks or jewelry (maximum 3% of image).')
    if isinstance(value,dict) and value.get('background') is not None:
        from .core import image_hash
        if image_hash(editor_photo(value))!=image_hash(target):
            raise ValueError('Spot cleanup belongs to a different target photo. Clear it or paint this target again.')
