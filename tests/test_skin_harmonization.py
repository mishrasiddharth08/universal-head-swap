import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from khs import core


def fixture(target_skin,generated_skin,background=(10,180,230)):
    size=(160,200)
    original=Image.new('RGB',size,background)
    ImageDraw.Draw(original).rectangle((50,132,110,195),fill=target_skin)
    mask=Image.new('L',size,0)
    ImageDraw.Draw(mask).ellipse((35,12,125,125),fill=255)
    mask=mask.filter(ImageFilter.GaussianBlur(5))
    generated=Image.new('RGB',size,generated_skin)
    draw=ImageDraw.Draw(generated)
    draw.rectangle((40,18,120,42),fill=(12,10,9))       # hair
    draw.rectangle((58,64,69,70),fill=(18,35,70))       # eyes
    draw.rectangle((91,64,102,70),fill=(18,35,70))
    # Fine luminance texture must survive exposure/chroma correction.
    pixels=generated.load()
    for y in range(75,112):
        for x in range(55,106):
            d=5 if (x+y)%2 else -5
            pixels[x,y]=tuple(max(0,min(255,c+d)) for c in generated_skin)
    return original,generated,core.EditRegion(original,(0,0,*size),mask,original)


class SkinHarmonizationTests(unittest.TestCase):
    def test_light_and_dark_tints_move_toward_local_neck_without_flattening(self):
        for target,tint in [((210,160,130),(245,130,105)),((92,61,45),(138,70,55))]:
            with self.subTest(target=target):
                original,generated,region=fixture(target,tint)
                raw=core.composite_region(generated,region,0)
                result=core.composite_region(generated,region,.5)
                target_array=np.array(target,dtype=float)
                before=np.linalg.norm(np.asarray(raw,dtype=float)[92,80]-target_array)
                after=np.linalg.norm(np.asarray(result,dtype=float)[92,80]-target_array)
                self.assertLess(after,before)
                raw_texture=np.asarray(raw,dtype=float)[75:112,55:106,0].std()
                corrected_texture=np.asarray(result,dtype=float)[75:112,55:106,0].std()
                self.assertGreater(corrected_texture,raw_texture*.75)

    def test_colored_background_excluded_and_features_preserved(self):
        original,generated,region=fixture((205,155,125),(240,125,100),(220,40,210))
        raw=core.composite_region(generated,region,0)
        result=core.composite_region(generated,region,.5)
        # Correction follows the local neck, not the magenta background.
        before=np.linalg.norm(np.asarray(raw,dtype=float)[92,80]-np.array((205,155,125)))
        after=np.linalg.norm(np.asarray(result,dtype=float)[92,80]-np.array((205,155,125)))
        self.assertLess(after,before)
        for point in ((80,25),(63,67),(96,67)):
            self.assertLessEqual(sum(abs(a-b) for a,b in zip(result.getpixel(point),raw.getpixel(point))),3)

    def test_local_skin_lighting_matches_without_transferring_source_detail(self):
        original,generated,region=fixture((205,155,125),(200,160,105))
        # A real face exists in the source, lit differently across each cheek.
        original=original.copy()
        draw=ImageDraw.Draw(original)
        draw.ellipse((35,12,125,125),fill=(220,155,140))
        draw.rectangle((78,45,112,112),fill=(180,120,108))
        region=core.EditRegion(original,region.box,region.mask,original)
        raw=core.composite_region(generated,region,0)
        result=core.composite_region(generated,region,.5)
        for x in (66,94):
            target=np.asarray(original,dtype=float)[92,x]
            before=np.linalg.norm(np.asarray(raw,dtype=float)[92,x]-target)
            after=np.linalg.norm(np.asarray(result,dtype=float)[92,x]-target)
            self.assertLess(after,before)
        # Generated high-frequency texture survives; no source feature copy.
        a=np.asarray(result,dtype=float)[80:105,57:73,0]
        self.assertGreater(np.abs(np.diff(a,axis=1)).mean(),5)
        outside=np.asarray(region.mask)==0
        np.testing.assert_array_equal(np.asarray(result)[outside],np.asarray(original)[outside])

    def test_bright_teeth_and_eye_whites_keep_generated_color(self):
        original,generated,region=fixture((205,155,125),(240,125,100))
        draw=ImageDraw.Draw(generated)
        draw.rectangle((58,64,69,70),fill=(230,230,225))
        draw.rectangle((74,88,87,94),fill=(242,235,220))
        raw=core.composite_region(generated,region,0)
        result=core.composite_region(generated,region,.5)
        for point in ((63,67),(80,91)):
            self.assertLessEqual(sum(abs(a-b) for a,b in zip(result.getpixel(point),raw.getpixel(point))),3)

    def test_exact_outside_mask_and_zero_strength(self):
        original,generated,region=fixture((205,155,125),(240,125,100))
        zero=core.composite_region(generated,region,0)
        manual=original.copy(); manual.paste(generated,(0,0))
        manual=Image.composite(manual,original,region.mask)
        np.testing.assert_array_equal(np.asarray(zero),np.asarray(manual))
        result=core.composite_region(generated,region,.5)
        outside=np.asarray(region.mask)==0
        np.testing.assert_array_equal(np.asarray(result)[outside],np.asarray(original)[outside])

    def test_inadequate_neck_sample_rejects_correction(self):
        original,generated,region=fixture((205,155,125),(240,125,100))
        original=Image.new('RGB',original.size,(10,180,230))
        region=core.EditRegion(original,region.box,region.mask,original)
        raw=core.composite_region(generated,region,0)
        result=core.composite_region(generated,region,.5)
        self.assertLessEqual(np.abs(np.asarray(result,dtype=int)-np.asarray(raw,dtype=int)).max(),1)


if __name__=='__main__': unittest.main()
