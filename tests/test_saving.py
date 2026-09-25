"""Actual JPEG overflow and lossless PNG metadata regression, CPU only."""
import io
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs.saving import protect_jpeg_metadata

class SavingTests(unittest.TestCase):
    def params(self, text, filename='grid.jpg'):
        session=NS(closed=False,owner=NS(last_report={}),host=NS(shared=NS(opts=NS(enable_pnginfo=True))))
        return NS(p=NS(_khs_session=session),filename=filename,pnginfo={'parameters':text})
    def test_actual_overflow_falls_back_and_preserves_pixels_and_metadata(self):
        text='large negative prompt, '*4000
        exif=piexif.dump({'Exif':{piexif.ExifIFD.UserComment:piexif.helper.UserComment.dump(text,encoding='unicode')}})
        im=Image.new('RGB',(16,16),(32,91,202)); jpeg=io.BytesIO(); im.save(jpeg,format='JPEG')
        with self.assertRaises(struct.error): piexif.insert(exif,jpeg.getvalue(),io.BytesIO())
        p=self.params(text); protect_jpeg_metadata(p)
        self.assertTrue(p.filename.endswith('.png')); self.assertEqual(p.pnginfo['parameters'],text)
        meta=PngImagePlugin.PngInfo(); meta.add_text('parameters',p.pnginfo['parameters'])
        out=io.BytesIO(); im.save(out,format='PNG',pnginfo=meta); out.seek(0)
        with Image.open(out) as saved:
            self.assertEqual(saved.info['parameters'],text); self.assertEqual(saved.tobytes(),im.tobytes())
    def test_normal_jpeg_unchanged(self):
        p=self.params('normal'); protect_jpeg_metadata(p); self.assertEqual(p.filename,'grid.jpg')
    def test_unicode_uses_encoded_byte_limit(self):
        p=self.params('\U0001f600'*18000,'GRID.JPEG'); protect_jpeg_metadata(p); self.assertTrue(p.filename.endswith('.png'))
    def test_unrelated_closed_disabled_and_png_unchanged(self):
        for kind in ('unrelated','closed','disabled','png'):
            p=self.params('x'*40000)
            if kind=='unrelated': p.p._khs_session=None
            if kind=='closed': p.p._khs_session.closed=True
            if kind=='disabled': p.p._khs_session.host.shared.opts.enable_pnginfo=False
            if kind=='png': p.filename='grid.png'
            before=p.filename; protect_jpeg_metadata(p); self.assertEqual(p.filename,before)
    def test_fallback_names_are_distinct(self):
        a=self.params('x'*40000); b=self.params('x'*40000)
        protect_jpeg_metadata(a); protect_jpeg_metadata(b); self.assertNotEqual(a.filename,b.filename)
