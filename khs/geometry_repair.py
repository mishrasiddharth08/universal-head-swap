"""Bounded final-frame uniform correction; never stretches facial aspect ratio."""
import math
from PIL import Image

def repair(image,region,target,generated,analyzer,core):
    before=core.geometry_report(target,generated,image.size,image.size)
    info={'applied':False,'before':before,'kind':'uniform protected final-frame alignment'}
    if before['geometry_target_met']:return image,info
    tx0,ty0,tx1,ty1=target['box'];gx0,gy0,gx1,gy1=generated['box']
    tw,th=tx1-tx0,ty1-ty0;gw,gh=gx1-gx0,gy1-gy0
    if min(tw,th,gw,gh)<8:info['reason']='Face too small for reliable correction';return image,info
    rw,rh=gw/tw,gh/th
    lower=max(.85,.90/rw,.92/rh);upper=min(1.15,1.10/rw,1.08/rh)
    if lower>upper:
        info.update(reason='Shape drift cannot meet width/height limits with uniform scaling; regenerate with model-specific pose conditioning',requires_resampling=True)
        return image,info
    scale=max(lower,min(upper,2/(rw+rh)))
    tx,gx=(tx0+tx1)/2,(gx0+gx1)/2
    if math.hypot(tx-gx,ty1-gy1)>th*.20:
        info['reason']='Correction would move the head too far';return image,info
    dx,dy=tx-scale*gx,ty1-scale*gy1
    affine=(1/scale,0,-dx/scale,0,1/scale,-dy/scale)
    moved=image.transform(image.size,Image.Transform.AFFINE,affine,resample=Image.Resampling.BICUBIC)
    valid=Image.new('L',image.size,255).transform(image.size,Image.Transform.AFFINE,affine,resample=Image.Resampling.NEAREST)
    moved=Image.composite(moved,image,valid)
    mask=region.mask
    if mask.size!=image.size:
        from .masking import resize_mask
        mask=resize_mask(mask,image.size)
    candidate=Image.composite(moved,image,mask)
    face=core.nearest_face(analyzer.faces(candidate),target,candidate.size)
    if not face:info['reason']='Face not measurable after correction';return image,info
    after=core.geometry_report(target,face,candidate.size,candidate.size)
    old_detail=core.metrics(core.face_crop(image,generated))['sharp'];new_detail=core.metrics(core.face_crop(candidate,face))['sharp']
    info.update(after=after,scale_x=scale,scale_y=scale,detail_before=old_detail,detail_after=new_detail)
    if not after['geometry_target_met'] or not core.alignment_improves(before,after,th):
        info['reason']='Final composite did not pass the geometry gate';return image,info
    if new_detail+1e-6<old_detail*.95:
        info['reason']='Correction reduced measured local detail';return image,info
    info['applied']=True
    return candidate,info
