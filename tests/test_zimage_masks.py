import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core,zimage

class ZImageMaskTests(unittest.TestCase):
    def tearDown(self):zimage.clear_mask_cache()
    def test_fast_mask_needs_no_optional_checkpoint(self):
        self.assertIsNone(zimage.prepare_mask(Image.new('RGB',(100,100)),None,core.normalize({}),None))
    def test_custom_mask_takes_priority_over_unavailable_sam3(self):
        mask=Image.new('L',(100,100));ImageDraw.Draw(mask).rectangle((20,20,40,50),fill=255)
        cfg=core.normalize({'zimage_mask_source':'SAM3 (optional)'})
        self.assertIs(zimage.prepare_mask(Image.new('RGB',mask.size),None,cfg,mask),mask)
        with self.assertRaisesRegex(ValueError,'detected face'):
            zimage.prepare_mask(Image.new('RGB',mask.size),None,cfg,None)
    def test_cpu_release_and_neck_protection_after_blur(self):
        class Fake:
            error=''
            def __init__(self,*a,**kw):self.device=kw['device'];self.released=0
            def mask(self,image,**kw):return Image.new('L',image.size,255)
            def release(self,**kw):self.released+=1
        image=Image.new('RGB',(160,180));pose={'box':(50,40,100,100),'head_w':50,'head_h':60,'head_px':60}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'sam3.pt';checkpoint.write_bytes(b'test')
            cfg=core.normalize({'zimage_mask_source':'SAM3 (optional)','sam3_checkpoint':str(checkpoint)})
            with patch('khs.sam3_mask.SAM3Masker',Fake):
                mask=zimage.prepare_mask(image,pose,cfg,None)
                self.assertEqual(zimage._MASKER.device,'cpu');self.assertEqual(zimage._MASKER.released,1)
                region=core.build_region(image,pose,feather=.25,mask=mask)
                region=zimage.protect_sam3_neck(region,pose,cfg)
                self.assertEqual(np.asarray(region.mask)[107:].max(),0)
                self.assertGreater(np.asarray(region.mask)[60:90,60:90].max(),0)
                cfg['custom_mask']=mask
                original=region.mask
                self.assertIs(zimage.protect_sam3_neck(region,pose,cfg).mask,original)
