"""Picker and uploaded-mask callbacks without loading Forge or Gradio."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from khs import core

def callback(name,env):
    tree=ast.parse((ROOT/'scripts/klein_face_reference.py').read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<ui callback>','exec'),env)
    return env[name]

class PickerTests(unittest.TestCase):
    def setUp(self):
        self.entries={n:SimpleNamespace(metadata={}) for n in ('z-image-char-alice','z-image-char-bob','z-image-lightning','bfs_head_v1_qwen_2.1','qwen-character')}
        self.env={'gr':SimpleNamespace(update=lambda **kw:kw),'core':core,'names':[],
                  'runtime':SimpleNamespace(registry=lambda:self.entries),'sam3_installed':lambda:None}
        self.sync=callback('sync_adapter',self.env)
        net=ModuleType('networks');net.list_available_networks=lambda:None
        self.netpatch=patch.dict(sys.modules,{'networks':net});self.netpatch.start();self.addCleanup(self.netpatch.stop)
    def test_identity_is_preserved_and_fresh_choices_used(self):
        result=self.sync('zimage','Z-Image-Turbo',True,'z-image-char-bob')
        self.assertEqual(result[4]['value'],'z-image-char-bob')
        self.assertIn('z-image-char-bob',result[4]['choices'])
        self.assertIn('z-image-char-bob',result[0]['choices'])
    def test_identity_is_never_guessed(self):
        self.assertEqual(self.sync('zimage','Z-Image-Turbo',True)[4]['value'],'None (skip)')
    def test_cross_family_identity_is_cleared(self):
        self.assertEqual(self.sync('qwen','Qwen-Image-Edit',True,'z-image-char-bob')[4]['value'],'None (skip)')
        self.assertEqual(self.sync('zimage','Z-Image-Turbo',True,'qwen-character')[4]['value'],'None (skip)')
    def test_matching_identity_is_preserved(self):
        self.assertEqual(self.sync('qwen','Qwen-Image-Edit',True,'qwen-character')[4]['value'],'qwen-character')
    def test_identity_clears_even_without_bfs(self):
        del self.entries['bfs_head_v1_qwen_2.1']
        self.assertEqual(self.sync('qwen','Qwen-Image-Edit',True,'z-image-char-bob')[4]['value'],'None (skip)')
    def test_manual_selection_is_unchanged(self):
        self.assertNotIn('value',self.sync('qwen','Qwen-Image-Edit',False,'z-image-char-bob')[4])
    def test_checkpoint_variant_wins_over_stale_preset(self):
        for preset,checkpoint,expected in [('zit','Z-Image-BF16.safetensors','Base'),('zib','ZIT-Q8.gguf','Turbo'),('zit','ZIB-BF16.safetensors','Base')]:
            for automatic in (True,False):
                result=self.sync(preset,checkpoint,automatic)
                self.assertEqual(result[1]['value'],expected)
                if not automatic: self.assertNotIn('value',result[0])
    def test_variant_mismatch_clears_character(self):
        self.entries['z-image-base-char']=SimpleNamespace(metadata={})
        for automatic in (True,False):
            result=self.sync('zit','Z-Image-Turbo',automatic,'z-image-base-char')
            self.assertEqual(result[4]['value'],'None (skip)')
    def test_model_matching_preserves_sam3_settings(self):
        self.env['sam3_installed']=lambda:'models/SAM 3/sam3.pt'
        result=self.sync('zit','Z-Image-Turbo',True)
        self.assertNotIn('value',result[2]);self.assertNotIn('value',result[3])
    def test_non_zimage_turbo_checkpoint_is_not_zimage(self):
        self.entries['bfs_head_swap_v1.1_krea2']=SimpleNamespace(metadata={})
        result=self.sync('krea2','Krea2-Turbo-BF16.safetensors',True)
        self.assertEqual(result[0]['value'],'bfs_head_swap_v1.1_krea2')
        self.assertNotIn('value',result[1])
        self.assertTrue(result[5]['visible'])
    def test_character_variant_checks_entry_filename(self):
        self.entries['identity']=SimpleNamespace(metadata={},filename='models/Lora/ZIB/person.safetensors')
        self.assertEqual(self.sync('zit','Z-Image-Turbo',True,'identity')[4]['value'],'None (skip)')
    def test_upload_becomes_a_layer_on_black(self):
        from PIL import Image
        upload=callback('import_mask',{})
        mask=Image.new('RGBA',(4,3),(128,128,128,255))
        result=upload(mask)
        self.assertEqual(result['background'].getpixel((0,0)),(0,0,0,255))
        self.assertEqual(result['layers'][0].getpixel((0,0)),(128,128,128,255))
        self.assertIs(result['composite'],result['layers'][0])
        self.assertIsNone(upload(None))

if __name__=='__main__': unittest.main()
