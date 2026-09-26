import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from PIL import Image
import test_runtime as fixtures
from khs import core, runtime
from khs.external import ExternalSession


class ExternalTests(unittest.TestCase):
    def session(self, count=13):
        _,host,p,owner,cfg=fixtures.RuntimeTests().fixtures()
        cfg['headshots']=[Image.new('RGB',(256,256),(i*10,100,100)) for i in range(count)]
        owner.custom={}; owner.choices=core.choices({})
        p.extra_generation_params={}; p.prompt='preserve the scene'; p.negative_prompt='blur'
        model=NS(text_processing_engine_qwen=object(),ref_latents=[],ini_latent=None)
        host.dynamic.edit=True
        with patch('khs.runtime.resolve_adapters',return_value=('bfs_head_v1_qwen_2.1','character')),patch('khs.runtime.registry',return_value={}):
            s=ExternalSession(p,cfg,owner,host,model=model)
        return s,p,host

    def test_thirteen_headshots_feed_only_selected_pair_and_restore_request(self):
        s,p,host=self.session()
        original=p.init_images; dimensions=(p.width,p.height); original_dynamic=host.dynamic.ref_latents
        try:
            s.enter()
            plan,refs=s.prepare(0,'<lora:turbo:1> preserve scene','blur',17,1,turbo=True)
            self.assertEqual(len(s.refs),13)
            self.assertEqual(len(refs),2)
            self.assertIn('<image1>',plan.positive); self.assertIn('<image2>',plan.positive)
            self.assertNotIn('(head_swap:',plan.positive)
            self.assertIn('<lora:turbo:1>',plan.positive)
            self.assertIn('<lora:bfs_head_v1_qwen_2.1:',plan.positive)
            self.assertIn('<lora:character:',plan.positive)
            self.assertEqual(plan.cfg,1)
            output=s.finish_image(Image.new('RGB',(512,512)),0)
            self.assertEqual(output.size,original[0].size)
        finally: s.close()
        self.assertIs(p.init_images,original)
        self.assertEqual((p.width,p.height),dimensions)
        self.assertEqual(p.batch_size,2)
        self.assertEqual(host.dynamic.ref_latents,original_dynamic)

    def test_reference_rotation_and_manual_slot(self):
        s,p,host=self.session()
        s.cfg['pick_mode']='Always rotation'
        try:
            s.enter()
            for i in (0,12,13):
                s.prepare(i,'swap','',i,1)
                self.assertEqual(s.selected_ref,i%13)
            s.cfg['manual_slot']='7'
            s.prepare(14,'swap','',14,1)
            self.assertEqual(s.selected_ref,6)
        finally: s.close()

    def test_exact_preferred_adapters_inside_subfolders(self):
        entries={'Z/bfs_head_v1_qwen_2.1':{},'A/bfs_head_v0_qwen_2.1':{},
                 'Z/bfs_head_v1_flux-klein_9b_step3500_rank128':{},'A/bfs_head_v1_flux-klein_9b':{}}
        self.assertEqual(core.select_adapter(entries,None,'qwen'),'Z/bfs_head_v1_qwen_2.1')
        self.assertEqual(core.select_adapter(entries,9),'Z/bfs_head_v1_flux-klein_9b_step3500_rank128')

    def test_auto_model_overrides_stale_dropdown_but_manual_mode_does_not(self):
        q='folder/bfs_head_v1_qwen_2.1'
        entries={q:NS(metadata={},filename=q+'.safetensors',alias=q)}
        cfg=core.normalize({'lora_dropdown':'bfs_head_v1_flux-klein_9b','auto_model_adapter':True})
        with patch('khs.runtime.registry',return_value=entries):
            self.assertEqual(runtime.resolve_adapters(cfg,NS(),'qwen'),(q,''))
            cfg['auto_model_adapter']=False
            with self.assertRaises(ValueError): runtime.resolve_adapters(cfg,NS(),'qwen')

    def test_protected_rgba_output_keeps_pixels_outside_mask(self):
        from PIL import ImageDraw
        s,p,host=self.session(1)
        s.cfg['edit_scope']='Protected head edit'
        mask=Image.new('L',s.original.size)
        ImageDraw.Draw(mask).rectangle((90,90,160,160),fill=255)
        s.cfg['custom_mask']=mask
        try:
            s.enter()
            s.prepare(0,'swap','',7,1)
            result=s.finish_image(Image.new('RGBA',s.external_canvas_size,(255,0,0,255)),0)
            self.assertEqual(result.size,s.original.size)
            self.assertEqual(result.getpixel((0,0)),s.original.getpixel((0,0)))
            self.assertGreater(result.getpixel((125,125))[0],200)
        finally: s.close()

    def test_turbo_tags_are_detected_without_matching_ordinary_text(self):
        self.assertTrue(core.has_speed_lora('<lora:folder/klein_9B_Turbo_r128:1>'))
        self.assertFalse(core.has_speed_lora('a turbo sports car <lora:character:0.7>'))

if __name__=='__main__': unittest.main()
