from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core,zimage


class ZImageIdentityTests(unittest.TestCase):
    def session(self,denoise):
        mask=Image.new('L',(80,80),0)
        ImageDraw.Draw(mask).ellipse((20,10,60,60),fill=255)
        values={}
        session=NS(
            region=NS(mask=mask,box=(0,0,80,80)),canvas_box=(0,0,80,80),
            canvas_size=(80,80),cfg={'zimage_denoise':denoise,
                                    'zimage_mask_source':'Face detector (fast)'},
            target_pose={'box':(20,10,60,60)},analysis={},
            host=NS(dynamic=NS(is_referencing=True,edit=True)),
            set_p=lambda key,value:values.__setitem__(key,value))
        zimage.configure_inpaint(session)
        return session,values

    def test_partial_denoise_keeps_valid_image_latent(self):
        session,values=self.session(0.65)
        self.assertEqual(values['inpainting_fill'],1)
        self.assertEqual(session.analysis['native_inpaint']['masked_content'],'original latent')
        self.assertEqual(set(np.unique(np.asarray(values['latent_mask']))),{0,255})
        self.assertTrue(values['mask_round'])
        self.assertFalse(session.host.dynamic.is_referencing)

    def test_full_denoise_can_start_from_noise(self):
        session,values=self.session(1.0)
        self.assertEqual(values['inpainting_fill'],2)
        self.assertEqual(session.analysis['native_inpaint']['masked_content'],'latent noise')


if __name__=='__main__': unittest.main()
