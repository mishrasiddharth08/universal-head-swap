"""Final processing preserves protected pixels and facial aspect ratio."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from khs import core, runtime


class ProtectedFinishing(unittest.TestCase):
    def session(self, family='klein', protected=True, **flags):
        s=runtime.Session.__new__(runtime.Session)
        s.cfg=core.normalize(dict(geometry_match=False, match_sharpness=False,
                                  identity_check=False, **flags))
        s.original=Image.new('RGB',(128,128),(80,100,120))
        s.target_pose=None
        mask=Image.new('L',s.original.size,0)
        ImageDraw.Draw(mask).rectangle((32,32,95,95),fill=255)
        s.region=core.EditRegion(s.original,(0,0,128,128),mask,s.original.copy()) if protected else None
        s.canvas_box=(0,0,128,128) if protected else None
        s.canvas_size=(128,128)
        s.owner=NS(analyzer=NS(faces=lambda im:[]),last_report={})
        s.quality_history=[]; s.family=family; s.selected_ref=None
        s.plans=[NS(seed=42)]
        s.p=NS(extra_generation_params={})
        s.host=NS(shared=NS(state=NS(textinfo='')))
        return s

    def test_global_finishing_preserves_black_mask_pixels(self):
        for flags in ({'sharpness':100}, {'hdr_enable':True,'hdr_gain':.8}, {'grain_amount':.1}):
            with self.subTest(flags=flags):
                s=self.session(**flags)
                generated=Image.fromarray(np.random.default_rng(2).integers(0,256,(128,128,3),dtype=np.uint8))
                result=np.asarray(s.finish_image(generated,0))
                outside=np.asarray(s.region.mask)==0
                self.assertTrue(np.array_equal(result[outside],np.asarray(s.original)[outside]))

    def test_full_image_finishing_remains_global(self):
        s=self.session(protected=False,hdr_enable=True,hdr_gain=.8)
        result=s.finish_image(s.original,0)
        self.assertNotEqual(result.getpixel((0,0)),s.original.getpixel((0,0)))

    def test_explicit_cleanup_can_edit_outside_head_mask(self):
        s=self.session(hdr_enable=True,hdr_gain=.8)
        s.cfg['cleanup_mask']=object()
        def repair(image,mask):
            image=image.copy(); image.putpixel((0,0),(255,0,0))
            return image,{'applied':True}
        with patch('khs.cleanup.applies',return_value=True),patch('khs.cleanup.repair',side_effect=repair):
            result=s.finish_image(s.original,0)
        self.assertEqual(result.getpixel((0,0)),(255,0,0))

    def test_all_native_routes_request_uniform_alignment(self):
        pose=dict(box=(40,40,80,80),head_w=40,head_h=40,head_px=40)
        for family in ('klein','qwen','krea','zimage'):
            with self.subTest(family=family):
                s=self.session(family); s.cfg['geometry_match']=True; s.target_pose=pose
                s.owner.analyzer.faces=lambda im:[pose]
                with patch('khs.core.align_protected_head',return_value=(s.original,{})) as align:
                    s.finish_image(s.original,0)
                self.assertIs(align.call_args.kwargs['preserve_aspect'],True)


if __name__=='__main__':
    unittest.main()
