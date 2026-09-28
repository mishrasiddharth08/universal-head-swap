import unittest
import numpy as np
from PIL import Image,ImageDraw
from khs.cleanup import repair,painted_mask

class CleanupTests(unittest.TestCase):
    def test_repairs_mark_without_changing_unpainted_pixels(self):
        im=Image.new('RGB',(100,100),(190,145,120));ImageDraw.Draw(im).line((48,45,48,54),fill='black',width=2)
        mask=Image.new('L',im.size);ImageDraw.Draw(mask).rectangle((46,43,51,56),fill=255)
        result,report=repair(im,mask);a=np.asarray(im);b=np.asarray(result);m=np.asarray(mask)>0
        self.assertTrue(np.array_equal(a[~m],b[~m]))
        self.assertGreater(b[48,48].mean(),100)
        self.assertEqual(result.size,im.size);self.assertTrue(report['applied'])
    def test_photo_background_is_never_a_mask(self):
        im=Image.new('RGB',(100,100),'white')
        self.assertIsNone(painted_mask({'background':im,'composite':im,'layers':[]},im.size))
    def test_only_white_painted_layer_is_selected(self):
        layer=Image.new('RGBA',(100,100));ImageDraw.Draw(layer).rectangle((20,20,24,24),fill='white')
        mask=painted_mask({'layers':[layer]},layer.size)
        self.assertEqual(mask.getbbox(),(20,20,25,25))
    def test_rejects_large_or_mismatched_mask(self):
        im=Image.new('RGB',(100,100))
        with self.assertRaisesRegex(ValueError,'3%'):repair(im,Image.new('L',im.size,255))
        with self.assertRaisesRegex(ValueError,'dimensions'):repair(im,Image.new('L',(50,50),255))
    def test_empty_cleanup_is_exact_bypass(self):
        im=Image.new('RGB',(100,100))
        out,report=repair(im,None);self.assertIs(out,im);self.assertFalse(report['applied'])

    def test_candidates_require_selection_and_match_original(self):
        from khs.cleanup import select_candidates,validate_target
        from khs.core import image_hash
        photo=Image.new('RGB',(100,100),(190,145,120));mask=Image.new('L',photo.size)
        ImageDraw.Draw(mask).rectangle((45,45,49,49),fill=255)
        state={'hash':image_hash(photo),'labels':['1: tattoo'],'proposals':[{'mask':mask}]}
        editor={'background':photo,'layers':[]}
        empty=select_candidates(editor,state,[])
        self.assertIsNone(painted_mask(empty,photo.size))
        selected=select_candidates(editor,state,['1: tattoo'])
        self.assertEqual(painted_mask(selected,photo.size).getbbox(),mask.getbbox())
        validate_target(selected,photo)
        with self.assertRaisesRegex(ValueError,'different target'):validate_target(selected,Image.new('RGB',photo.size,'black'))
        with self.assertRaisesRegex(ValueError,'changed'):select_candidates({'background':Image.new('RGB',photo.size)},state,['1: tattoo'])
